"""
Usage:
    streamlit run Backend-RAG/run_TAAassistant_page.py
"""
import os
os.environ["USE_VLLM"] = "0"
import sys
import logging
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Force spawn before torch import to avoid CUDA re-init after fork.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import torch
import torch.multiprocessing as torch_mp
import streamlit as st
from utils import get_embeddings  # noqa: E402

# Align torch.multiprocessing with spawn; some libs consult torch_mp directly.
try:
    torch_mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Hint vLLM to use spawn if supported. vLLM itself is disabled here via USE_VLLM=0.
os.environ.setdefault("VLLM_WORKER_MULTIPROCESSING_START_METHOD", "spawn")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from run_TAassistant import (  # noqa: E402
    build_quiz_chains,
    generate_quiz,
    grade_quiz_answer,
    retrieval_qa_pipline,
    load_model,
)
from constants import MODEL_ID, MODEL_BASENAME  # noqa: E402


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def stream_text(text: str, chunk_size: int = 32):
    """
    Yield the response text in small chunks so Streamlit can render a streaming-like experience.
    """
    for idx in range(0, len(text), chunk_size):
        yield text[idx : idx + chunk_size]


def maybe_rewrite(rewriter, text: str, submode: str | None = None) -> str:
    """
    Apply query rewriting when a rewriter is provided; otherwise return the original text.
    """
    return rewriter.rewrite(text, submode=submode) if rewriter else text


@st.cache_resource(show_spinner=True)
def load_llm(device_type: str):
    """
    Load and cache the LLM so it can be reused across mode switches without reloading to GPU.
    """
    return load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)


@st.cache_resource(show_spinner=True)
def load_embeddings(device_type: str):
    """Cache embeddings per device to avoid reloading on mode switch."""
    if device_type == "hpu":
        from gaudi_utils.embeddings import load_embeddings as load_emb

        return load_emb()
    return get_embeddings(device_type)


@st.cache_resource(show_spinner=True)
def load_chain(
    device_type: str,
    use_history: bool,
    model_type: str,
    mode: str,
    enable_rewrite: bool,
    enable_web_search: bool,
    web_k: int,
    local_score_threshold: float,
):
    llm = load_llm(device_type)
    embeddings = load_embeddings(device_type)
    logging.info(
        (
            "Initializing QA chain: device=%s, history=%s, model=%s, mode=%s, rewrite=%s, "
            "web=%s (k=%s, thresh=%.2f)"
        ),
        device_type,
        use_history,
        model_type,
        mode,
        enable_rewrite,
        enable_web_search,
        web_k,
        local_score_threshold,
    )
    if mode == "quiz":
        question_chain, grading_chain, rewriter = build_quiz_chains(
            device_type=device_type,
            use_history=use_history,
            promptTemplate_type=model_type,
            llm=llm,
            embeddings=embeddings,
        )
        return {
            "mode": "quiz",
            "question_chain": question_chain,
            "grading_chain": grading_chain,
            "rewriter": rewriter if enable_rewrite else None,
        }

    qa_chain, rewriter = retrieval_qa_pipline(
        device_type=device_type,
        use_history=use_history,
        promptTemplate_type=model_type,
        mode=mode,
        enable_rewrite=enable_rewrite,
        enable_web_search=enable_web_search,
        web_k=web_k,
        local_relevance_threshold=local_score_threshold,
        llm=llm,
        embeddings=embeddings,
    )
    return {"mode": mode, "chain": qa_chain, "rewriter": rewriter if enable_rewrite else None}


def main():
    st.set_page_config(page_title="TA Assistant for Database Systems", page_icon="💬", layout="wide")
    st.title("TA Assistant for Database Systems Course")

    with st.sidebar:
        st.header("Settings")
        device_type = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)
        resolved_device = pick_device() if device_type == "auto" else device_type
        model_type = st.selectbox("model", ["qwen2.5", "llama3", "llama", "mistral", "non_llama"], index=0)
        mode = st.selectbox("Mode", ["answer", "hint", "quiz"], index=0)
        use_history = st.checkbox("Use chat history", value=False)
        show_sources = st.checkbox("Show sources", value=True)
        enable_rewrite = st.checkbox("Enable query rewriting", value=True)
        enable_web_search = st.checkbox("Enable web search (answer mode only)", value=True)
        web_k = st.slider("Web results to merge", min_value=1, max_value=8, value=3, step=1)
        local_score_threshold = st.slider(
            "Local relevance threshold (0-1)", min_value=0.0, max_value=1.0, value=0.25, step=0.05
        )

    if "qa_config" not in st.session_state:
        st.session_state.qa_config = {}
    if "qa_bundle" not in st.session_state:
        st.session_state.qa_bundle = None
    if "quiz_state" not in st.session_state:
        st.session_state.quiz_state = {"questions": [], "docs": [], "topic": "", "retrieval_topic": "", "feedback": []}

    cfg = {
        "device_type": resolved_device,
        "use_history": use_history,
        "model_type": model_type,
        "mode": mode,
        "enable_rewrite": enable_rewrite,
        "enable_web_search": enable_web_search,
        "web_k": web_k,
        "local_score_threshold": float(local_score_threshold),
    }

    if st.session_state.qa_config != cfg or st.session_state.qa_bundle is None:
        st.session_state.qa_bundle = load_chain(**cfg)
        st.session_state.qa_config = cfg
        st.session_state.quiz_state = {
            "questions": [],
            "docs": [],
            "topic": "",
            "retrieval_topic": "",
            "feedback": [],
        }
        for key in [k for k in st.session_state.keys() if k.startswith("quiz_ans_")]:
            st.session_state.pop(key)

    bundle = st.session_state.qa_bundle

    if bundle["mode"] == "quiz":
        quiz_show_sources = False  # always hide sources in quiz mode
        quiz_topic = st.text_area(
            "Enter a quiz topic/request",
            value=st.session_state.quiz_state.get("topic", ""),
            height=120,
            placeholder="e.g., Generate some quizzes for unit 15 query optimization.",
        )
        generate = st.button("Generate quiz", type="primary")

        if generate:
            if not quiz_topic.strip():
                st.warning("Please enter a quiz topic to generate questions.")
            else:
                with st.spinner("Generating quiz questions..."):
                    questions, docs, retrieval_topic = generate_quiz(
                        bundle["question_chain"],
                        quiz_topic,
                        rewriter=bundle.get("rewriter"),
                    )
                questions = questions[:3]  # double-guard to keep exactly 3
                st.session_state.quiz_state = {
                    "questions": questions,
                    "docs": docs,
                    "topic": quiz_topic,
                    "retrieval_topic": retrieval_topic,
                    "feedback": [],
                }
                for idx in range(len(questions)):
                    st.session_state[f"quiz_ans_{idx}"] = ""

        questions = st.session_state.quiz_state.get("questions", [])
        if questions:
            st.subheader("Quiz Questions")
            for idx, question in enumerate(questions, start=1):
                st.markdown(f"**{idx}. {question}**")
            retrieval_topic = st.session_state.quiz_state.get("retrieval_topic", "")
            if bundle.get("rewriter") and retrieval_topic and retrieval_topic != st.session_state.quiz_state.get("topic"):
                st.caption(f"Retrieval topic: {retrieval_topic}")
            if quiz_show_sources and st.session_state.quiz_state.get("docs"):
                st.subheader("Sources")
                for idx, document in enumerate(st.session_state.quiz_state["docs"], start=1):
                    meta = document.metadata or {}
                    source = meta.get("source") or "unknown source"
                    unit_no = meta.get("unit_no") or meta.get("unit_id")
                    source_files = meta.get("source_files")

                    title = source
                    if unit_no:
                        title += f" | unit={unit_no}"
                    if source_files:
                        title += f" | files={source_files}"

                    st.markdown(f"**{idx}. {title}**")
                    st.write(document.page_content)
                    st.divider()

            st.subheader("Your answers")
            answers = []
            for idx, question in enumerate(questions):
                ans_key = f"quiz_ans_{idx}"
                ans_val = st.text_area(f"Answer for Q{idx + 1}", key=ans_key, height=80)
                answers.append(ans_val.strip())

            grade = st.button("Grade answers")
            if grade:
                feedback_entries = [None] * len(questions)
                rewriter = bundle.get("rewriter")

                def _grade_one(idx: int, question: str, user_ans: str):
                    if not user_ans:
                        return idx, {
                            "question": question,
                            "feedback": "No answer provided. Skipped grading.",
                            "docs": [],
                        }
                    feedback_text, grade_docs, _ = grade_quiz_answer(
                        bundle["grading_chain"],
                        question,
                        user_ans,
                        rewriter=rewriter,
                    )
                    return idx, {"question": question, "feedback": feedback_text, "docs": grade_docs}

                max_workers = min(4, len(questions)) or 1
                with st.spinner("Grading answers..."):
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(_grade_one, idx, question, answers[idx])
                            for idx, question in enumerate(questions)
                        ]
                        for fut in as_completed(futures):
                            idx, entry = fut.result()
                            feedback_entries[idx] = entry

                st.session_state.quiz_state["feedback"] = feedback_entries

            feedback = st.session_state.quiz_state.get("feedback") or []
            if feedback:
                st.subheader("Feedback")
                for idx, fb in enumerate(feedback, start=1):
                    st.markdown(f"**Q{idx}: {fb['question']}**")
                    st.write(fb["feedback"])
                    if quiz_show_sources and fb.get("docs"):
                        st.markdown("**Sources**")
                        for doc in fb["docs"]:
                            meta = doc.metadata or {}
                            source = meta.get("source") or "unknown source"
                            unit_no = meta.get("unit_no") or meta.get("unit_id")
                            source_files = meta.get("source_files")

                            title = source
                            if unit_no:
                                title += f" | unit={unit_no}"
                            if source_files:
                                title += f" | files={source_files}"

                            st.markdown(f"- **{title}**")
                            st.write(doc.page_content)
                        st.divider()

    else:
        user_query = st.text_area("Ask a question", value="", height=120, placeholder="What is the two-phase locking?")
        ask = st.button("Ask", type="primary")

        if ask and user_query.strip():
            with st.spinner("Thinking..."):
                rewriter = bundle.get("rewriter")
                retrieval_query = maybe_rewrite(rewriter, user_query)
                res = bundle["chain"].invoke(retrieval_query)
            answer = res.get("result") or ""
            docs = res.get("source_documents") or []

            st.subheader("Answer")
            answer_text = answer if isinstance(answer, str) else str(answer)
            st.write_stream(stream_text(answer_text))
            if rewriter and retrieval_query != user_query:
                st.caption(f"Retrieval query: {retrieval_query}")

            if show_sources and docs:
                st.subheader("Sources")
                for idx, document in enumerate(docs, start=1):
                    meta = document.metadata or {}
                    source = meta.get("source", "unknown source")
                    st.markdown(f"**{idx}. {source}**")
                    st.write(document.page_content)
                    st.divider()

            if mode != "answer" and enable_web_search:
                st.info("Web search is only applied in answer mode; skipped for this mode.")
        elif ask:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
