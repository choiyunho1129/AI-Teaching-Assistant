import logging
import os
import re
import click
import torch
import utils
from typing import Any, Callable, List, Optional, Tuple
from langchain_classic.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.memory import ConversationBufferMemory
try:
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
except ImportError:
    DuckDuckGoSearchAPIWrapper = None
from langchain_chroma import Chroma
import torchvision
torchvision.disable_beta_transforms_warning()

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import (
    extract_rewritten,
    get_rewrite_prompt,
    get_prompt_template,
    get_system_prompt,
    get_quiz_question_prompt_template,
    get_quiz_grading_prompt_template,
)
from utils import get_embeddings
from transformers import pipeline
from DB import parse_unit_from_filename
from load_models import (
    load_quantized_model_awq, load_quantized_model_gguf_ggml,
    load_quantized_model_qptq, load_full_model
)
from constants import (
    EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, HINT_PERSIST_DIRECTORY, QUIZ_PERSIST_DIRECTORY,
    ANSWER_COLLECTION_NAME, HINT_COLLECTION_NAME, QUIZ_COLLECTION_NAME,
    MODEL_ID, MODEL_BASENAME, MAX_NEW_TOKENS, CHROMA_SETTINGS
)

os.environ["USE_VLLM"] = "0"

class QueryRewriter:
    """
    Uses the *current* LLM (same one used for answering) to rewrite queries for retrieval.
    Different rewrite guidelines per mode/submode.
    """
    def __init__(self, llm, mode: str):
        self.llm = llm
        self.mode = mode

    def rewrite(self, query: str, submode: Optional[str] = None) -> str:
        prompt = get_rewrite_prompt(self.mode, submode=submode).format(query=query)

        try:
            res = self.llm.invoke(prompt)
            text = res.content if hasattr(res, "content") else str(res)
            rewritten = extract_rewritten(text)
            return rewritten if rewritten else query
        except Exception:
            return query


def _extract_numbered_questions(block: str) -> List[str]:
    """
    Parse a blob of numbered questions (1., 2), etc.) into a clean list.
    Falls back to returning the whole block if numbering is missing.
    """
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    questions: List[str] = []
    current: List[str] = []
    for ln in lines:
        m = re.match(r"^\s*\d+[\.\)]\s*(.*)", ln)
        if m:
            if current:
                questions.append(" ".join(current).strip())
            current = [m.group(1).strip()]
        else:
            if current:
                current.append(ln)
            else:
                current = [ln]
    if current:
        questions.append(" ".join(current).strip())
    return questions or [block.strip()]

class ChromaEmbeddingFunction:
    """
    Adapter for LangChain Embeddings -> Chroma EmbeddingFunction.
    Provides embed_documents/embed_query (used by LangChain) and __call__(input) (used by Chroma client).
    """

    def __init__(self, embedder):
        self.embedder = embedder

    def embed_documents(self, texts):
        return self.embedder.embed_documents(texts)

    def embed_query(self, text):
        return self.embedder.embed_query(text)

    def __call__(self, input):
        if isinstance(input, str):
            return [self.embed_query(input)]
        return self.embed_documents(list(input))


class SimpleChromaRetriever(BaseRetriever):
    """Minimal retriever to satisfy BaseRetriever interface for newer LangChain."""

    vectorstore: Any
    k: int = 4
    add_source_to_content: bool = False
    query_hint_func: Optional[Callable[[str], str]] = None
    web_search: Optional[Callable[[str, int], List[Document]]] = None
    web_k: int = 3
    relevance_score_threshold: Optional[float] = None
    unit_filter_fields: Tuple[str, ...] = ("unit_no", "unit_id")
    unit_hint_extractor: Optional[Callable[[str], List[str]]] = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query, run_manager=None):
        search_query = query
        # 0) quiz에서 unit 힌트가 있으면 metadata filter로 먼저 고정 검색
        unit_hints = []
        if self.unit_hint_extractor:
            unit_hints = self.unit_hint_extractor(query)

        def _similarity_search_with_filter(q: str, k: int, flt: dict) -> List[Document]:
            # langchain_chroma 버전에 따라 filter= or where= 차이가 있어 방어적으로 처리
            try:
                return self.vectorstore.similarity_search(q, k=k, filter=flt)
            except TypeError:
                return self.vectorstore.similarity_search(q, k=k, where=flt)

        if unit_hints:
            # unit_no는 "03" 형태로 들어가게 ingest에서 zfill 권장했으니, 여기서도 보정
            normalized = []
            for u in unit_hints:
                u = str(u).strip()
                if u.isdigit():
                    normalized.append(u.zfill(2))
                else:
                    normalized.append(u)

            collected: List[Document] = []
            for u in normalized:
                for field in self.unit_filter_fields:
                    try:
                        sub = _similarity_search_with_filter(query, self.k, {field: u})
                        if sub:
                            collected.extend(sub)
                            break  # field 하나로 잡히면 다음 unit으로
                    except Exception:
                        continue
                if len(collected) >= self.k:
                    break

            if collected:
                # 중복 제거 후 상위 k개
                seen = set()
                uniq = []
                for d in collected:
                    key = ((d.metadata or {}).get("id")) or (d.page_content[:200])
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(d)
                docs = uniq[: self.k]
            else:
                docs = self.vectorstore.similarity_search(query, k=self.k)

        else:
            # 기존 로직: hint string 덧붙이기는 quiz에서는 쓰지 않는 게 안정적
            search_query = self.query_hint_func(query) if self.query_hint_func else query
            docs = self.vectorstore.similarity_search(search_query, k=self.k)

        # 빈 결과거나 점수가 낮으면 웹 검색을 붙인다.
        if self.web_search:
            need_web = not docs
            if not need_web and self.relevance_score_threshold is not None:
                try:
                    scored = self.vectorstore.similarity_search_with_relevance_scores(search_query, k=self.k)
                    if scored:
                        docs = [doc for doc, _ in scored]
                        max_score = max(score for _, score in scored)
                        if max_score < self.relevance_score_threshold:
                            need_web = True
                except Exception:
                    # 점수 검색 실패 시 그냥 기존 docs만 사용
                    pass
            if need_web:
                try:
                    web_docs = self.web_search(query, self.web_k)
                except Exception:
                    web_docs = []
                if web_docs:
                    docs = (docs or []) + web_docs

        # (이하 웹검색/annotate 로직은 그대로 두되, quiz에서는 web_search=None이라 영향 없음)
        if not self.add_source_to_content:
            return docs

        annotated_docs = []
        for doc in docs:
            metadata = doc.metadata or {}
            source = metadata.get("source", "unknown source")
            page = metadata.get("page") or metadata.get("page_number") or metadata.get("page_no")
            slide = metadata.get("slide") or metadata.get("slide_number") or metadata.get("slide_no")
            loc = source
            if page is not None:
                loc += f" (page {page})"
            if slide is not None:
                loc += f" (slide {slide})"
            doc.page_content = f"[Source: {loc}]\n{doc.page_content}"
            annotated_docs.append(doc)
        return annotated_docs


_UNIT_HINT_PATTERNS = [
    re.compile(r"(?:unit|lecture|chapter|week|단원|강|주차)\s*(\d{1,2})", re.IGNORECASE),
    re.compile(r"#\s*(\d{1,2})"),
    re.compile(r"\b(\d{1,2})\s*(?:강|차시|번)\b"),
    re.compile(r"\b(\d{1,2})[_-]?#\b"),
]


def extract_unit_hints(query: str) -> List[str]:
    """
    Try to pull out unit numbers or file-based prefixes mentioned in the user query.
    Accepts patterns like '03_#03', '#03', '3강', 'unit 3', etc.
    """
    hints: List[str] = []
    direct_unit, _ = parse_unit_from_filename(query.strip())
    if direct_unit:
        hints.append(direct_unit)

    for token in re.split(r"[\s,;]+", query):
        if not token:
            continue
        unit_no, _ = parse_unit_from_filename(token)
        if unit_no:
            hints.append(unit_no)

    for pat in _UNIT_HINT_PATTERNS:
        for m in pat.finditer(query):
            hints.append(m.group(1))

    deduped: List[str] = []
    seen = set()
    for h in hints:
        candidates = [h]
        stripped = h.lstrip("0")
        if stripped:
            candidates.append(stripped)
        if stripped.isdigit() and len(stripped) == 1:
            candidates.append(stripped.zfill(2))

        for c in candidates:
            if not c or c in seen:
                continue
            seen.add(c)
            deduped.append(c)
    return deduped


def generate_quiz(
    question_chain,
    quiz_topic: str,
    rewriter: Optional[QueryRewriter] = None,
) -> Tuple[List[str], List[Document], str]:
    """
    Run quiz question generation and return parsed questions, source documents, and the retrieval topic used.
    Returns the retrieval topic actually used (useful when rewriting is enabled).
    """
    retrieval_topic = rewriter.rewrite(quiz_topic, submode="quiz_generation") if rewriter else quiz_topic
    quiz_res = question_chain.invoke(retrieval_topic)
    questions_blob, docs = quiz_res["result"], quiz_res["source_documents"]
    questions = _extract_numbered_questions(questions_blob)
    # Enforce at most 3 questions to avoid runaway generations
    if len(questions) > 3:
        questions = questions[:3]
    return questions, docs, retrieval_topic


def grade_quiz_answer(
    grading_chain,
    question: str,
    user_answer: str,
    rewriter: Optional[QueryRewriter] = None,
) -> Tuple[str, List[Document], str]:
    """
    Grade a single quiz answer and return feedback, supporting docs, and the retrieval query used.
    """
    grading_query = (
        f"Quiz question: {question}\nStudent answer: {user_answer}\n"
        "Judge correctness using the retrieved context. Provide verdict, correct answer, and brief explanation."
    )
    # No rewriting for grading to avoid altering the question/answer semantics.
    grade_res = grading_chain.invoke(grading_query)
    return grade_res["result"], grade_res.get("source_documents", []), grading_query


def run_quiz_cli(question_chain, grading_chain, show_sources: bool, rewriter: Optional[QueryRewriter] = None) -> None:
    while True:
        query = input("\nEnter a quiz topic/request (or 'exit' to quit): ")
        if query == "exit":
            break

        questions, docs, retrieval_topic = generate_quiz(question_chain, query, rewriter=rewriter)

        print("\n\n> Quiz request:")
        print(query)
        if rewriter and retrieval_topic != query:
            print("\n> Retrieval topic (rewritten):")
            print(retrieval_topic)

        print("\n\n> Quiz Questions (no answers):")
        for idx, q in enumerate(questions, start=1):
            print(f"{idx}. {q}")

        if show_sources and docs:
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                md = document.metadata or {}
                src = md.get("source") or "unknown source"
                unit_no = md.get("unit_no") or md.get("unit_id")
                source_files = md.get("source_files")

                title = src
                if unit_no:
                    title += f" | unit={unit_no}"
                if source_files:
                    title += f" | files={source_files}"

                print("\n> " + title + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")

        for idx, q in enumerate(questions, start=1):
            user_ans = input(f"\nYour answer for Q{idx} (type 'skip' to move on or 'exit' to quit): ").strip()
            if user_ans.lower() == "exit":
                return
            if user_ans.lower() == "skip":
                continue

            feedback, grade_docs, retrieval_grading = grade_quiz_answer(
                grading_chain, q, user_ans, rewriter=rewriter
            )

            print(f"\n> Feedback for Q{idx}:")
            print(feedback)

            if show_sources and grade_docs:
                print("----------------------------------SOURCE DOCUMENTS---------------------------")
                for document in grade_docs:
                    print("\n> " + document.metadata.get("source", "unknown source") + ":")
                    print(document.page_content)
                print("----------------------------------SOURCE DOCUMENTS---------------------------")


def run_qa_cli(qa, mode: str, show_sources: bool, save_qa: bool, rewriter: Optional[QueryRewriter] = None) -> None:
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        retrieval_query = rewriter.rewrite(query) if rewriter else query

        res = qa.invoke(retrieval_query)
        answer, docs = res["result"], res["source_documents"]

        print("\n\n> Question:")
        print(query)
        if rewriter and retrieval_query != query:
            print("\n> Retrieval query (rewritten):")
            print(retrieval_query)

        label = "Hint" if mode == "hint" else "Answer"
        print(f"\n> {label}:")
        print(answer)

        if show_sources:
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")

        if save_qa:
            utils.log_to_csv(query, answer)


def make_duckduckgo_search_fn(default_k: int = 3) -> Optional[Callable[[str, int], List[Document]]]:
    """
    Create a small wrapper that fetches DuckDuckGo search results and turns them into Documents.
    Returns None if the dependency is missing so the caller can gracefully skip web search.
    """
    if DuckDuckGoSearchAPIWrapper is None:
        logging.warning("DuckDuckGoSearchAPIWrapper not available; web search disabled.")
        return None

    search_client = DuckDuckGoSearchAPIWrapper()

    def _search(query: str, k: Optional[int] = None) -> List[Document]:
        num_results = k or default_k
        try:
            # Try several known signatures to stay compatible across versions.
            attempts = [
                lambda: search_client.results(query, max_results=num_results),
                lambda: search_client.results(query, num_results=num_results),
                lambda: search_client.results(query, k=num_results),
                lambda: search_client.results(query, num_results),
                lambda: search_client.results(query),
            ]
            results = None
            for attempt in attempts:
                try:
                    results = attempt()
                    break
                except TypeError:
                    continue
            if results is None:
                raise TypeError("DuckDuckGoSearchAPIWrapper.results signature mismatch")
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            logging.warning(f"Web search failed ({exc}); continuing without web context.")
            return []

        docs: List[Document] = []
        for item in results:
            url = item.get("link") or item.get("href") or ""
            title = item.get("title") or url or "web result"
            snippet = item.get("snippet") or item.get("body") or ""
            content = f"{title}\n{snippet}".strip()
            docs.append(
                Document(
                    page_content=content,
                    metadata={"source": url or "web", "type": "web_search", "title": title},
                )
            )
        return docs

    return _search


def build_quiz_chains(
    device_type: str,
    use_history: bool,
    promptTemplate_type: str = "qwen",
    llm=None,
    embeddings=None,
):
    """
    Build two RetrievalQA chains for quiz mode:
    - question_chain: generates 3 questions without answers.
    - grading_chain: checks a user's answer and provides verdict/correct answer/explanation.
    """
    if embeddings is None:
        if device_type == "hpu":
            from gaudi_utils.embeddings import load_embeddings

            embeddings = load_embeddings()
        else:
            embeddings = get_embeddings(device_type)

    embedding_fn = ChromaEmbeddingFunction(embeddings)
    db = Chroma(
        persist_directory=QUIZ_PERSIST_DIRECTORY,
        embedding_function=embedding_fn,
        client_settings=CHROMA_SETTINGS,
        collection_name=QUIZ_COLLECTION_NAME,
    )
    retriever = SimpleChromaRetriever(
        vectorstore=db,
        k=4,
        add_source_to_content=False,
        query_hint_func=None,                 # quiz에서는 끄는 게 보통 더 안정
        unit_hint_extractor=extract_unit_hints,
        unit_filter_fields=("unit_no", "unit_id"),
    )

    llm = llm or load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    question_prompt = get_quiz_question_prompt_template()
    grading_prompt = get_quiz_grading_prompt_template()

    question_chain_kwargs = {"prompt": question_prompt}
    grading_chain_kwargs = {"prompt": grading_prompt}

    if use_history:
        # Quiz 모드는 자체 프롬프트를 쓰므로, 대화 이력 공유만 위해 메모리만 생성한다.
        memory = ConversationBufferMemory(input_key="question", memory_key="history")
        question_chain_kwargs["memory"] = memory
        grading_chain_kwargs["memory"] = memory

    question_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        callbacks=callback_manager,
        chain_type_kwargs=question_chain_kwargs,
    )

    grading_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        callbacks=callback_manager,
        chain_type_kwargs=grading_chain_kwargs,
    )

    # ADD
    rewriter = QueryRewriter(llm, mode="quiz_generation")
    return question_chain, grading_chain, rewriter

def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")
    use_vllm_env = os.getenv("USE_VLLM", "").lower() in ("1", "true", "yes")

    # Use vLLM only when explicitly requested via USE_VLLM and when using a full HF model (no quantized basename).
    if use_vllm_env and model_basename is None:
        try:
            from langchain_community.llms import VLLM

            logging.info("Using vLLM for inference (USE_VLLM env enabled)")
            engine_kwargs = {
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.6,
            }
            sampling_kwargs = {
                "temperature": 0.7,
                "top_p": 0.8,
            }

            # Pin download_dir so weights are reused (no repeated downloads when the cache is persistent).
            hf_home = os.getenv("HF_HOME")
            download_dir = (
                os.getenv("VLLM_DOWNLOAD_DIR")
                or os.getenv("HUGGINGFACE_HUB_CACHE")
                or os.getenv("TRANSFORMERS_CACHE")
                or (os.path.join(hf_home, "hub") if hf_home else None)
                or os.path.expanduser("~/.cache/huggingface/hub")
            )

            logging.info(
                "vLLM engine params (applied): max_model_len=%s, gpu_memory_utilization=%s, download_dir=%s",
                engine_kwargs["max_model_len"],
                engine_kwargs["gpu_memory_utilization"],
                download_dir,
            )

            llm = VLLM(
                model=model_id,
                trust_remote_code=True,
                max_new_tokens=MAX_NEW_TOKENS,
                download_dir=download_dir,
                vllm_kwargs=engine_kwargs,   # forwarded to vLLM engine init
                model_kwargs=sampling_kwargs,  # forwarded to sampling params
                # also pass sampling params at top-level for older versions
                **sampling_kwargs,
            )
            return llm
        except Exception as exc:
            logging.warning(f"Falling back to transformers pipeline (vLLM unavailable): {exc}")
    
    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        elif ".awq" in model_basename.lower():
            model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Avoid HF warning about missing pad token by aligning pad/eos.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    if getattr(model, "config", None) is not None and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Create a pipeline for text generation
    if device_type == "hpu":
        from gaudi_utils.pipeline import GaudiTextGenerationPipeline

        pipe = GaudiTextGenerationPipeline(
            model_name_or_path=model_id,
            max_new_tokens=1000,
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.15,
            do_sample=True,
            max_padding_length=5000,
        )
        pipe.compile_graph()
    else:
        pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.15,
        return_full_text=False,  # only return the assistant reply, not the prompt
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


def retrieval_qa_pipline(
    device_type,
    use_history,
    promptTemplate_type="qwen",
    mode="answer",
    enable_web_search=True,
    web_k=3,
    local_relevance_threshold=0.25,
    enable_rewrite=True,
    llm=None,
    embeddings=None,
):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQA: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    """
    (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within ingest.py.

    (2) Provides additional arguments for instructor and BGE models to improve results, pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    """
    if device_type == "hpu":
        from gaudi_utils.embeddings import load_embeddings

        embeddings = embeddings or load_embeddings()
    else:
        embeddings = embeddings or get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    # load the vectorstore
    embedding_fn = ChromaEmbeddingFunction(embeddings)
    if mode == "hint":
        persist_dir = HINT_PERSIST_DIRECTORY
        collection_name = HINT_COLLECTION_NAME
    elif mode == "quiz":
        persist_dir = QUIZ_PERSIST_DIRECTORY
        collection_name = QUIZ_COLLECTION_NAME
    else:
        persist_dir = PERSIST_DIRECTORY
        collection_name = ANSWER_COLLECTION_NAME
    if mode == "quiz" and not os.path.isdir(persist_dir):
        logging.warning(
            f"Quiz DB not found at {persist_dir}. Run Ingest_specific_DB.py with --build quiz to populate it."
        )
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_fn,
        client_settings=CHROMA_SETTINGS,
        collection_name=collection_name,
    )
    retriever = SimpleChromaRetriever(
        vectorstore=db,
        k=4,
        add_source_to_content=(mode == "hint"),
        query_hint_func=None,
        web_search=make_duckduckgo_search_fn(default_k=web_k) if enable_web_search and mode == "answer" else None,
        web_k=web_k,
        relevance_score_threshold=local_relevance_threshold if enable_web_search and mode == "answer" else None,
    )

    # get the prompt template and memory if set by the user.
    system_prompt = get_system_prompt(mode=mode)
    prompt, memory = get_prompt_template(
        system_prompt=system_prompt, promptTemplate_type=promptTemplate_type, history=use_history
    )

    # load the llm pipeline (allow preloaded llm to be reused)
    llm = llm or load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )
    rewriter = QueryRewriter(llm, mode=mode) if enable_rewrite else None

    return qa, rewriter


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    is_flag=True,
    help="Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="qwen2.5",
    type=click.Choice(
        ["llama3", "llama", "mistral", "qwen2.5", "non_llama"],
    ),
    help="model type: llama3, llama, mistral, qwen2.5 or non_llama",
)
@click.option(
    "--mode",
    default="answer",
    type=click.Choice(["answer", "hint", "quiz"]),
    help="answer: normal Q&A, hint: point to materials without the solution, quiz: generate 3 key quiz items for a unit.",
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="whether to save Q&A pairs to a CSV file (Default is False)",
)
@click.option(
    "--enable_web_search/--disable_web_search",
    default=True,
    show_default=True,
    help="When in answer mode, fetch web snippets if local similarity is low or empty.",
)
@click.option(
    "--web_k",
    default=3,
    show_default=True,
    help="How many web search snippets to merge when web search is triggered.",
)
@click.option(
    "--local_score_threshold",
    default=0.25,
    show_default=True,
    help="Min relevance score (0-1) from the vector store required to skip web search.",
)
@click.option(
    "--enable_rewrite/--disable_rewrite",
    default=True,
    show_default=True,
    help="Rewrite queries for retrieval using the current LLM (mode-specific guidelines).",
)

def main(
    device_type,
    show_sources,
    use_history,
    model_type,
    mode,
    save_qa,
    enable_web_search,
    web_k,
    local_score_threshold,
    enable_rewrite,
):
    """
    Implements the main information retrieval task for a localGPT.

    This function sets up the QA system by loading the necessary embeddings, vectorstore, and LLM model.
    It then enters an interactive loop where the user can input queries and receive answers. Optionally,
    the source documents used to derive the answers can also be displayed.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'mps', 'cuda', etc.
    - show_sources (bool): Flag to determine whether to display the source documents used for answering.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Notes:
    - Logging information includes the device type, whether source documents are displayed, and the use of history.
    - If the models directory does not exist, it creates a new one to store models.
    - The user can exit the interactive loop by entering "exit".
    - The source documents are displayed if the show_sources flag is set to True.

    """

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")
    logging.info(f"Mode set to: {mode}")
    if enable_web_search and mode != "answer":
        logging.warning("Web search flag is only applied in answer mode; ignoring for current mode.")
    logging.info(f"Web search enabled: {enable_web_search if mode == 'answer' else False}")

    if mode == "quiz":
        question_chain, grading_chain, rewriter = build_quiz_chains(
            device_type=device_type,
            use_history=use_history,
            promptTemplate_type=model_type,
        )
        # NOTE: quiz rewrite uses the same LLM; different submodes for generation/grading
        run_quiz_cli(question_chain, grading_chain, show_sources, rewriter if enable_rewrite else None)
    else:
        qa, rewriter = retrieval_qa_pipline(
            device_type,
            use_history,
            promptTemplate_type=model_type,
            mode=mode,
            enable_web_search=enable_web_search,
            web_k=web_k,
            local_relevance_threshold=local_score_threshold,
            enable_rewrite=enable_rewrite,
        )
        run_qa_cli(qa, mode, show_sources, save_qa, rewriter)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
