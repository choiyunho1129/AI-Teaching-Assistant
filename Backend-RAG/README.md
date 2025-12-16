# TA Assistant for Database Systems Course

RAG(Retrieval-Augmented Generation) 기반 AI Teaching Assistant 시스템입니다. 데이터베이스 시스템 과목 학습을 위한 세 가지 모드(Answer, Hint, Quiz)를 제공합니다.

## 핵심 기능

| 모드 | 설명 | 사용 시나리오 |
|------|------|--------------|
| **Answer** | 질문에 대한 직접적인 답변 제공 | 개념 설명, 문제 해결 |
| **Hint** | 답변 대신 참조 위치 안내 | 자기주도 학습 지원 |
| **Quiz** | 퀴즈 생성 및 채점 | 학습 평가, 복습 |

## 프로젝트 구조

```
Backend-RAG/
├── run_TAassistant.py          # 핵심 QA 파이프라인, CLI 진입점
├── run_TAassistant_page.py     # Streamlit 웹 인터페이스
├── DB.py                       # 문서 로딩, 청킹, 벡터 DB 구축
├── load_models.py              # LLM 로딩 (Full/GGUF/GPTQ/AWQ)
├── prompt_template_utils.py    # 프롬프트 템플릿, Query Rewriting
├── utils.py                    # 임베딩 로딩, Q&A 로깅
├── constants.py                # 설정값, 경로, 모델 ID
│
├── SOURCE_DOCUMENTS/           # 강의 자료 (PDF, PPTX, DOCX)
├── class_transcript/           # YouTube 자막 텍스트
│
├── answer_DB/                  # Answer 모드 벡터 DB
├── hint_DB/                    # Hint 모드 벡터 DB
├── quiz_DB/                    # Quiz 모드 벡터 DB
│
└── local_chat_history/         # Q&A 로그 저장
    └── qa_log.csv
```

## 설치

### 1. 환경 요구사항

- Python 3.10+
- CUDA 지원 GPU (권장: VRAM 24GB 이상)

### 2. 의존성 설치

```bash
# 핵심 라이브러리
pip install langchain-core langchain-community langchain-classic langchain-chroma
pip install langchain-huggingface langchain-text-splitters
pip install transformers torch chromadb streamlit click

# 양자화 모델용 (선택)
pip install auto-gptq llama-cpp-python bitsandbytes

# 웹 검색용 (선택)
pip install duckduckgo-search

# 문서 로딩용 (선택)
pip install pypdf unstructured python-pptx
```

### 3. 강의 자료 준비

`SOURCE_DOCUMENTS/` 폴더에 강의 자료를 배치합니다. 파일명은 다음 형식을 권장합니다:

```
01_#01 - Relational Model.pdf
02_#02 - SQL Basics.pptx
...
```

YouTube 자막은 `class_transcript/` 폴더에 배치합니다:

```
01_#01 - Relational Model_7NPIENPr-zk.txt
...
```

## 실행 방법

### Step 1: 벡터 데이터베이스 구축

프로젝트를 처음 실행하거나 강의 자료가 변경된 경우, 먼저 벡터 DB를 구축해야 합니다.

```bash
# 모든 DB 구축 (answer, hint, quiz)
python DB.py --device_type cuda --build all

# 특정 DB만 구축
python DB.py --device_type cuda --build answer
python DB.py --device_type cuda --build hint
python DB.py --device_type cuda --build quiz
```

### Step 2-A: CLI로 실행

```bash
# Answer 모드 (기본)
python run_TAassistant.py --device_type cuda --mode answer --show_sources

# Hint 모드
python run_TAassistant.py --device_type cuda --mode hint

# Quiz 모드
python run_TAassistant.py --device_type cuda --mode quiz
```

**추가 옵션:**

```bash
# 웹 검색 비활성화
python run_TAassistant.py --disable_web_search

# Query Rewriting 비활성화
python run_TAassistant.py --disable_rewrite

# Q&A 로깅 활성화
python run_TAassistant.py --save_qa
```

### Step 2-B: 웹 인터페이스로 실행

```bash
streamlit run run_TAassistant_page.py
```

브라우저에서 `http://localhost:8501`로 접속합니다.

## 설정 변경

`constants.py`에서 주요 설정을 변경할 수 있습니다:

```python
# 모델 설정
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
MODEL_ID = "Qwen/Qwen2.5-32B-Instruct"

# 생성 파라미터
CONTEXT_WINDOW_SIZE = 4096
MAX_NEW_TOKENS = 1024
```

## 지원 파일 형식

| 형식 | 확장자 |
|------|--------|
| PDF | `.pdf` |
| PowerPoint | `.ppt`, `.pptx` |
| Word | `.doc`, `.docx` |
| Excel | `.xls`, `.xlsx` |
| 텍스트 | `.txt`, `.md` |
| 웹 | `.html` |
| 코드 | `.py` |
| 데이터 | `.csv` |

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.
