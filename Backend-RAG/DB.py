import logging
import os
import re
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import click
import torch

# -------------------------
# LangChain Document / Splitter / Vectorstore (버전 호환)
# -------------------------
try:
    # LangChain 0.1+ 계열
    from langchain_core.documents import Document
except ImportError:
    # 구버전 호환
    from langchain.schema import Document

try:
    from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

# Loader imports (버전별 경로 차이)
try:
    from langchain_community.document_loaders import UnstructuredPDFLoader
except ImportError:
    try:
        from langchain.document_loaders import UnstructuredPDFLoader
    except ImportError:
        UnstructuredPDFLoader = None  # type: ignore

try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    try:
        from langchain.document_loaders import PyPDFLoader
    except ImportError:
        PyPDFLoader = None  # type: ignore

try:
    from langchain_community.document_loaders import UnstructuredPowerPointLoader
except ImportError:
    try:
        from langchain.document_loaders import UnstructuredPowerPointLoader
    except ImportError:
        UnstructuredPowerPointLoader = None  # type: ignore

from utils import get_embeddings
from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP as BASE_DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
)

# -------------------------
# Utilities
# -------------------------
def file_log(logentry: str):
    with open("file_ingest.log", "a", encoding="utf-8") as f:
        f.write(logentry + "\n")
    print(logentry)


def _safe_relpath(path: str, base_dir: str) -> str:
    try:
        return os.path.relpath(path, base_dir)
    except Exception:
        return path


def _is_under_dir(path: str, base_dir: str) -> bool:
    try:
        ap = os.path.abspath(path)
        bd = os.path.abspath(base_dir)
        return os.path.commonpath([ap, bd]) == bd
    except Exception:
        return False


def _detect_source_type(file_path: str, transcript_dir: str, docs_dir: str) -> str:
    ap = os.path.abspath(file_path)
    if transcript_dir and _is_under_dir(ap, transcript_dir):
        return "transcript"
    if docs_dir and _is_under_dir(ap, docs_dir):
        return "source_document"
    return "unknown"


_YTID_SUFFIX = re.compile(r"_[A-Za-z0-9_-]{11}$")


def parse_unit_from_filename(file_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    input:
      01_#01 - Relational Model & Algebra (CMU ...)_7NPIENPr-zk.txt
      03_#03 - Database Storage_ Files, Pages, Tuples.pdf
      10_#10 - Recovery (ARIES).pptx

    output:
      (unit_no, unit_title)
    """
    stem = os.path.splitext(file_name)[0].strip()
    stem = _YTID_SUFFIX.sub("", stem).strip()

    if " - " not in stem:
        m = re.match(r"^(?P<unit>\d+)", stem)
        return (m.group("unit") if m else None, None)

    prefix, title_part = stem.split(" - ", 1)

    m = re.match(r"^(?P<unit>\d+)", prefix.strip())
    unit_no = m.group("unit") if m else None

    unit_title = title_part.split("(", 1)[0].strip()
    return unit_no, unit_title


def _seconds_to_ts(sec: int) -> str:
    hh = sec // 3600
    mm = (sec % 3600) // 60
    ss = sec % 60
    if hh > 0:
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    return f"{mm:02d}:{ss:02d}"


def _choose_loader_class(file_path: str):
    """
    For hint DB,
    PDF/PPTX의 '페이지/슬라이드' 메타데이터가 살아있게 하려면
    UnstructuredFileLoader(파일 통째 1문서)보다 '페이지/슬라이드 분해 로더'를 우선한다.
    """
    ext = os.path.splitext(file_path)[1].lower()

    # PDF: PyPDFLoader(페이지 단위) 우선, 없으면 UnstructuredPDFLoader(mode="elements")로 대체
    if ext == ".pdf":
        if PyPDFLoader is not None:
            return PyPDFLoader
        if UnstructuredPDFLoader is not None:
            return UnstructuredPDFLoader

    # PPT/PPTX: 슬라이드 단위 분해 로더가 있으면 사용
    if ext in [".ppt", ".pptx"]:
        if UnstructuredPowerPointLoader is not None:
            return UnstructuredPowerPointLoader

    # 그 외는 constants의 BASE_DOCUMENT_MAP 사용
    return BASE_DOCUMENT_MAP.get(ext)


def _build_loader(file_path: str, loader_class):
    """
    로더별로 kwargs가 달라서 여기서 통일적으로 생성.
    """
    ext = os.path.splitext(file_path)[1].lower()

    # PyPDFLoader는 기본 생성만으로 페이지 단위 문서 생성
    if PyPDFLoader is not None and loader_class is PyPDFLoader:
        return loader_class(file_path)

    # UnstructuredPDFLoader는 mode="elements"가 페이지 번호/요소 메타가 잘 붙는 편
    if UnstructuredPDFLoader is not None and loader_class is UnstructuredPDFLoader:
        return loader_class(file_path, mode="elements")

    # UnstructuredPowerPointLoader도 elements 모드가 슬라이드 메타에 유리
    if UnstructuredPowerPointLoader is not None and loader_class is UnstructuredPowerPointLoader:
        return loader_class(file_path, mode="elements")

    # 기본 생성
    return loader_class(file_path)


def normalize_page_slide_meta(doc: Document, file_path: str) -> None:
    """
    loader마다 page/slide 메타데이터 키가 달라서 표준화한다.

    표준 키:
      - ref_kind: pdf/ppt/doc/video 등
      - page_no: (pdf) 1-based
      - slide_no: (ppt/pptx) 1-based
    """
    ext = os.path.splitext(file_path)[1].lower()
    if doc.metadata is None:
        doc.metadata = {}
    md = doc.metadata

    if ext == ".pdf":
        md["ref_kind"] = "pdf"
        # 후보 키들 (로더/버전 차이)
        for k in ["page", "page_number", "pageIndex", "pagenumber", "page_num"]:
            if k in md:
                try:
                    p = int(md[k])
                    # PyPDFLoader는 보통 0-based page를 쓰는 경우가 많음 -> 1-based로 보정
                    md["page_no"] = p + 1 if p >= 0 and k in ["page", "pageIndex"] else p
                except Exception:
                    pass
                break

        # Unstructured(elements)는 metadata에 page_number가 흔함
        if "page_no" not in md and "page_number" in md:
            try:
                md["page_no"] = int(md["page_number"])
            except Exception:
                pass

    elif ext in [".ppt", ".pptx"]:
        md["ref_kind"] = "ppt"
        # Unstructured PPT loader는 보통 page_number(=slide)로 들어오는 경우가 많음
        if "page_number" in md and "slide_no" not in md:
            try:
                md["slide_no"] = int(md["page_number"])
            except Exception:
                pass
        for k in ["slide", "slide_number", "slideIndex", "pagenumber", "page"]:
            if k in md and "slide_no" not in md:
                try:
                    s = int(md[k])
                    md["slide_no"] = s + 1 if s >= 0 and k in ["slide", "slideIndex", "page"] else s
                except Exception:
                    pass
                break
    else:
        md.setdefault("ref_kind", "doc")


def enrich_metadata(doc: Document, file_path: str, base_dir: str, source_type: str) -> Document:
    """
    - hint/quiz 네비게이션을 위한 표준 메타데이터 주입
    - 단원 번호/제목(unit_no/unit_title) 및 grouping key(unit_key) 주입
    - PDF/PPTX 페이지/슬라이드 번호 표준화(page_no/slide_no)
    """
    if doc.metadata is None:
        doc.metadata = {}

    doc.metadata.setdefault("source", file_path)
    doc.metadata["source_type"] = source_type
    doc.metadata["file_name"] = os.path.basename(file_path)
    doc.metadata["rel_path"] = _safe_relpath(file_path, base_dir)

    # transcript filename pattern:
    if source_type == "transcript":
        m = re.match(r"^(?P<lecture>\d+).*_(?P<yid>[A-Za-z0-9_-]{11})\.txt$", doc.metadata["file_name"])
        if m:
            doc.metadata["lecture_no"] = m.group("lecture")
            doc.metadata["youtube_id"] = m.group("yid")

    # unit_no / unit_title (both transcripts & source docs)
    unit_no, unit_title = parse_unit_from_filename(doc.metadata["file_name"])
    if unit_no:
        doc.metadata["unit_no"] = unit_no
    if unit_title:
        doc.metadata["unit_title"] = unit_title

    if unit_no and unit_title:
        doc.metadata["unit_key"] = f"{unit_no} - {unit_title}"
    elif unit_no:
        doc.metadata["unit_key"] = unit_no
    elif unit_title:
        doc.metadata["unit_key"] = unit_title

    normalize_page_slide_meta(doc, file_path)
    return doc


# -------------------------
# Loading
# -------------------------
def load_single_document(file_path: str) -> Optional[List[Document]]:
    """
    loader.load()가 여러 Document를 반환하는 경우(PDF pages, PPT slides 등)
    전부 유지하여 hint DB에서 '어느 페이지/슬라이드' 참조가 가능하게 한다.
    """
    try:
        loader_class = _choose_loader_class(file_path)
        if not loader_class:
            file_log(f"{file_path} document type is undefined.")
            raise ValueError(f"Document type is undefined: {os.path.splitext(file_path)[1]}")

        file_log(f"{file_path} loaded.")
        loader = _build_loader(file_path, loader_class)
        loaded = loader.load()
        if not loaded:
            return None

        # 각 doc에 source 강제 주입(로더/버전별로 없을 수 있음)
        for d in loaded:
            if d.metadata is None:
                d.metadata = {}
            d.metadata.setdefault("source", file_path)

        return loaded

    except Exception as ex:
        file_log(f"{file_path} loading error:\n{ex}")
        return None


def load_document_batch(filepaths: List[str]) -> Tuple[List[Document], List[str]]:
    logging.info("Loading document batch")
    if not filepaths:
        return ([], [])

    docs: List[Document] = []
    with ThreadPoolExecutor(max_workers=max(len(filepaths), 1)) as exe:
        futures = [exe.submit(load_single_document, p) for p in filepaths]
        for f in futures:
            loaded_list = f.result()
            if loaded_list:
                docs.extend([d for d in loaded_list if d is not None])

    return (docs, filepaths)


def collect_paths(source_dir: str) -> List[str]:
    paths: List[str] = []
    if not source_dir or not os.path.isdir(source_dir):
        return paths

    # ingestable 확장자는 BASE_DOCUMENT_MAP + (ppt/pptx/pdf 강제 지원)
    ingestable_exts = set(BASE_DOCUMENT_MAP.keys()) | {".ppt", ".pptx", ".pdf"}

    for root, _, files in os.walk(source_dir):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in ingestable_exts:
                paths.append(os.path.join(root, file_name))
    return paths


def load_documents_from_dirs(*dirs: str) -> List[Document]:
    """
    - 전체 파일 목록 수집
    - ProcessPoolExecutor 시도
    - 실패 시 ThreadPoolExecutor fallback
    """
    paths: List[str] = []
    for d in dirs:
        paths.extend(collect_paths(d))

    if not paths:
        logging.warning("No ingestable files found.")
        return []

    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = max(1, round(len(paths) / n_workers))

    def _run_with_process_pool() -> List[Document]:
        docs: List[Document] = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i in range(0, len(paths), chunksize):
                batch = paths[i : i + chunksize]
                futures.append(executor.submit(load_document_batch, batch))

            for fut in as_completed(futures):
                contents, _ = fut.result()
                docs.extend([d for d in contents if d is not None])
        return docs

    def _run_with_thread_pool() -> List[Document]:
        docs: List[Document] = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i in range(0, len(paths), chunksize):
                batch = paths[i : i + chunksize]
                futures.append(executor.submit(load_document_batch, batch))

            for fut in as_completed(futures):
                contents, _ = fut.result()
                docs.extend([d for d in contents if d is not None])
        return docs

    try:
        return _run_with_process_pool()
    except Exception as ex:
        file_log(f"[WARN] ProcessPool failed, falling back to ThreadPool. Reason:\n{ex}")
        try:
            return _run_with_thread_pool()
        except Exception as ex2:
            file_log(f"[ERROR] ThreadPool also failed:\n{ex2}")
            return []


def split_documents_by_extension(documents: List[Document]) -> Tuple[List[Document], List[Document]]:
    text_docs, python_docs = [], []
    for doc in documents:
        if doc is None:
            continue
        src = doc.metadata.get("source", "")
        ext = os.path.splitext(src)[1].lower()
        if ext == ".py":
            python_docs.append(doc)
        else:
            text_docs.append(doc)
    return text_docs, python_docs


# -------------------------
# Hint-mode transcript segmentation (video timestamp 네비게이션)
# -------------------------
_TS_PATTERN = re.compile(r"(?P<ts>\b\d{1,2}:\d{2}(?::\d{2})?\b)")  # 1:23 or 01:23:45


def _ts_to_seconds(ts: str) -> Optional[int]:
    parts = ts.split(":")
    try:
        if len(parts) == 2:
            mm, ss = parts
            return int(mm) * 60 + int(ss)
        if len(parts) == 3:
            hh, mm, ss = parts
            return int(hh) * 3600 + int(mm) * 60 + int(ss)
    except Exception:
        return None
    return None


def segment_transcript_by_timestamps(doc: Document, window_sec: int = 60) -> List[Document]:
    """
    Transcript를 window_sec 단위로 segment 생성.
    각 segment는 start_ts/end_ts를 metadata에 넣는다.
    """
    text = doc.page_content or ""
    matches = list(_TS_PATTERN.finditer(text))
    if len(matches) < 1:
        return [doc]

    items: List[Tuple[int, str]] = []
    for i, m in enumerate(matches):
        ts = m.group("ts")
        sec = _ts_to_seconds(ts)
        if sec is None:
            continue

        start_idx = m.start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start_idx:end_idx].strip()
        if chunk:
            items.append((sec, chunk))

    if not items:
        return [doc]

    buckets: "OrderedDict[int, List[str]]" = OrderedDict()
    for sec, chunk in items:
        bstart = (sec // window_sec) * window_sec
        buckets.setdefault(bstart, []).append(chunk)

    out: List[Document] = []
    base_meta = dict(doc.metadata or {})
    for bstart, parts in buckets.items():
        bend = bstart + window_sec
        new_meta = dict(base_meta)
        new_meta["start_sec"] = bstart
        new_meta["end_sec"] = bend
        new_meta["start_ts"] = _seconds_to_ts(bstart)
        new_meta["end_ts"] = _seconds_to_ts(bend)
        out.append(Document(page_content="\n".join(parts).strip(), metadata=new_meta))

    return out


# -------------------------
# Building DBs
# -------------------------
def build_chroma(
    texts: List[Document],
    embeddings,
    persist_directory: str,
    collection_name: str,
):
    os.makedirs(persist_directory, exist_ok=True)
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=persist_directory,
        client_settings=CHROMA_SETTINGS,
        collection_name=collection_name,
    )
    db.persist()
    return db


# -------------------------
# Answer chunks
# -------------------------
def make_answer_chunks(documents: List[Document]) -> List[Document]:
    text_documents, python_documents = split_documents_by_extension(documents)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    )

    texts = text_splitter.split_documents(text_documents)
    texts.extend(python_splitter.split_documents(python_documents))
    return texts


# -------------------------
# Hint chunks (정확한 "어디를 봐야 하는지" 메타데이터 강화)
# -------------------------
def annotate_chunk_position(chunks: List[Document]) -> List[Document]:
    """
    같은 (source + page_no/slide_no + ref_kind) 단위로 chunk를 묶고,
    페이지/슬라이드 내부에서 chunk 순서를 매긴다.
    """
    groups = defaultdict(list)
    for d in chunks:
        md = d.metadata or {}
        key = (
            md.get("source", ""),
            md.get("ref_kind", ""),
            md.get("page_no", None),
            md.get("slide_no", None),
        )
        groups[key].append(d)

    out: List[Document] = []
    for _, docs in groups.items():
        docs_sorted = sorted(docs, key=lambda x: (x.metadata or {}).get("start_index", 0))
        total = len(docs_sorted)
        for i, d in enumerate(docs_sorted, 1):
            d.metadata["chunk_idx_in_ref"] = i
            d.metadata["chunks_in_ref"] = total
            out.append(d)
    return out


def add_hint_headers(chunks: List[Document]) -> List[Document]:
    """
    검색 안정성을 위해 chunk 텍스트 앞에 네비게이션 헤더를 주입한다.
    (모델이 답을 생성하는 대신, 참조 위치를 말하는 데에 최적)
    """
    out: List[Document] = []
    for d in chunks:
        md = d.metadata or {}
        unit = md.get("unit_key") or ""
        fn = md.get("file_name") or ""
        rk = md.get("ref_kind") or md.get("source_type") or ""

        loc = []
        if md.get("page_no"):
            loc.append(f"page={md['page_no']}")
        if md.get("slide_no"):
            loc.append(f"slide={md['slide_no']}")
        if md.get("start_ts") and md.get("end_ts"):
            loc.append(f"time={md['start_ts']}~{md['end_ts']}")
        if md.get("chunk_idx_in_ref") and md.get("chunks_in_ref"):
            loc.append(f"chunk={md['chunk_idx_in_ref']}/{md['chunks_in_ref']}")

        header = f"[HINT] unit={unit} file={fn} kind={rk} " + " ".join(loc)
        d.page_content = header.strip() + "\n" + (d.page_content or "")
        out.append(d)
    return out


def make_hint_chunks(documents: List[Document]) -> List[Document]:
    """
    Hint mode 목표:
      - 답을 직접 말하지 말고
      - "어디를 보면 되는지"를 매우 정확히 안내

    구현:
      - transcript: timestamp window segmentation + small chunking
      - docs(pdf/pptx): 페이지/슬라이드 단위 문서(로드 단계에서 이미 분해됨) + small chunking
      - add_start_index=True로 페이지/슬라이드 내 위치 추적
      - chunk_idx_in_ref/chunks_in_ref 추가
      - 헤더를 텍스트에 주입하여 검색/출력 품질 강화
    """
    hint_docs: List[Document] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100, add_start_index=True
    )

    for d in documents:
        if d is None:
            continue
        st = d.metadata.get("source_type", "unknown")

        if st == "transcript":
            segs = segment_transcript_by_timestamps(d)
            hint_docs.extend(splitter.split_documents(segs))
        else:
            hint_docs.extend(splitter.split_documents([d]))

    hint_docs = annotate_chunk_position(hint_docs)
    hint_docs = add_hint_headers(hint_docs)
    return hint_docs


# -------------------------
# Quiz chunks
# -------------------------
def make_quiz_chunks(documents: List[Document]) -> List[Document]:
    groups = defaultdict(list)
    for d in documents:
        if d is None:
            continue

        unit_no = (d.metadata or {}).get("unit_no")
        if unit_no and str(unit_no).isdigit():
            gid = str(unit_no).zfill(2)             # <- 핵심: unit_no 기반
        else:
            gid = (d.metadata or {}).get("unit_key") or (d.metadata or {}).get("file_name") or "unknown"

        groups[gid].append(d)

    unit_level_docs: List[Document] = []
    for gid, docs in groups.items():
        docs_sorted = sorted(docs, key=lambda x: ((x.metadata or {}).get("source_type",""), (x.metadata or {}).get("file_name","")))

        # unit_no/title 대표값
        unit_no = None
        unit_title = None
        for dd in docs_sorted:
            if not unit_no and (dd.metadata or {}).get("unit_no"):
                unit_no = str((dd.metadata or {}).get("unit_no"))
            if not unit_title and (dd.metadata or {}).get("unit_title"):
                unit_title = (dd.metadata or {}).get("unit_title")
        if unit_no and unit_no.isdigit():
            unit_no = unit_no.zfill(2)

        merged_parts = []
        source_files = []
        for dd in docs_sorted:
            fn = (dd.metadata or {}).get("file_name", "unknown")
            source_files.append(fn)
            merged_parts.append(f"[FILE: {fn} | TYPE: {(dd.metadata or {}).get('source_type','unknown')}]")
            merged_parts.append(dd.page_content or "")

        header = [f"UNIT_NO: {unit_no}" if unit_no else "", f"UNIT_TITLE: {unit_title}" if unit_title else "", f"UNIT_ID: {gid}"]
        merged_text = ("\n".join([h for h in header if h]).strip() + "\n\n" + "\n".join(merged_parts).strip()).strip()

        base_meta = {
            "doc_role": "quiz_context",
            "unit_id": gid,
            "unit_no": unit_no,
            "unit_title": unit_title,
            "unit_key": f"{unit_no} - {unit_title}" if unit_no and unit_title else (unit_no or unit_title or gid),
            "source": f"unit:{gid}",
            "source_files": ", ".join(source_files),
        }
        unit_level_docs.append(Document(page_content=merged_text, metadata=base_meta))

    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=250)
    return splitter.split_documents(unit_level_docs)

# -------------------------
# CLI
# -------------------------
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        ["cpu","cuda","mps","xpu","hip","hpu","mtia","meta","mkldnn","opencl","vulkan","xla","lazy","ort","ipu","fpga","ve","opengl","ideep"]
    ),
)
@click.option("--source_documents_dir", default="./Backend-RAG/SOURCE_DOCUMENTS", show_default=True, help="Slides/PDF/docs directory")
@click.option("--class_transcript_dir", default="./Backend-RAG/class_transcript", show_default=True, help="YouTube transcript directory")
@click.option("--persist_answer_dir", default="answer_DB", show_default=True)
@click.option("--persist_hint_dir", default="hint_DB", show_default=True)
@click.option("--persist_quiz_dir", default="quiz_DB", show_default=True)
@click.option("--build", type=click.Choice(["answer", "hint", "quiz", "all"]), default="all", show_default=True)
def main(
    device_type: str,
    source_documents_dir: str,
    class_transcript_dir: str,
    persist_answer_dir: str,
    persist_hint_dir: str,
    persist_quiz_dir: str,
    build: str,
):
    logging.info(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = get_embeddings(device_type)

    logging.info(f"Loading documents from: {source_documents_dir}, {class_transcript_dir}")
    raw_docs = load_documents_from_dirs(source_documents_dir, class_transcript_dir)

    # Enrich metadata (unit/page/slide/timestamp 등)
    enriched: List[Document] = []
    for d in raw_docs:
        if d is None:
            continue
        src = d.metadata.get("source", "")
        st = _detect_source_type(src, class_transcript_dir, source_documents_dir)
        base_dir = class_transcript_dir if st == "transcript" else source_documents_dir
        enriched.append(enrich_metadata(d, src, base_dir, st))

    logging.info(f"Loaded {len(enriched)} documents (raw, incl. pages/slides).")

    if build in ("answer", "all"):
        answer_chunks = make_answer_chunks(enriched)
        logging.info(f"[AnswerDB] Split into {len(answer_chunks)} chunks.")
        build_chroma(
            texts=answer_chunks,
            embeddings=embeddings,
            persist_directory=persist_answer_dir,
            collection_name="answer_db",
        )
        logging.info(f"[AnswerDB] persisted at: {persist_answer_dir}")

    if build in ("hint", "all"):
        hint_chunks = make_hint_chunks(enriched)
        logging.info(f"[HintDB] Split into {len(hint_chunks)} chunks.")
        build_chroma(
            texts=hint_chunks,
            embeddings=embeddings,
            persist_directory=persist_hint_dir,
            collection_name="hint_db",
        )
        logging.info(f"[HintDB] persisted at: {persist_hint_dir}")

    if build in ("quiz", "all"):
        quiz_chunks = make_quiz_chunks(enriched)
        logging.info(f"[QuizDB] Split into {len(quiz_chunks)} chunks.")
        build_chroma(
            texts=quiz_chunks,
            embeddings=embeddings,
            persist_directory=persist_quiz_dir,
            collection_name="quiz_db",
        )
        logging.info(f"[QuizDB] persisted at: {persist_quiz_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    main()
