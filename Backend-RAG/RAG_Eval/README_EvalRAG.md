# TA Assistant RAG System Evaluation

이 패키지는 데이터베이스 시스템 과목용 TA Assistant RAG 시스템을 체계적으로 평가하기 위한 도구입니다.

## 📁 파일 구조

```
evaluation/
├── test_dataset.json      # 30개 테스트 Q&A 데이터셋
├── evaluate_rag.py        # 메인 평가 스크립트
├── analyze_results.py     # 결과 분석 및 시각화
├── requirements.txt       # 의존성 목록
└── README.md             # 이 문서
```

## 🚀 빠른 시작

### 1. 의존성 설치

```bash
pip install openai matplotlib pandas
# 또는 Anthropic API 사용 시
pip install anthropic matplotlib pandas
```

### 2. API 키 설정

```bash
# OpenAI 사용 시
export OPENAI_API_KEY="your-api-key"

# Anthropic 사용 시
export ANTHROPIC_API_KEY="your-api-key"
```

### 3. 평가 실행

#### Standalone 테스트 (TA 시스템 없이 평가 로직 테스트)
```bash
python evaluate_rag.py --standalone --dataset test_dataset.json --output results/test.json
```

#### 단일 설정 평가
```bash
# Baseline (기능 없음)
python evaluate_rag.py --config baseline --output results/baseline.json

# Full (모든 기능 활성화)
python evaluate_rag.py --config full --output results/full.json
```

#### Ablation Study (전체 비교)
```bash
python evaluate_rag.py --ablation --output results/ablation_study.json
```

### 4. 결과 분석

```bash
python analyze_results.py --input results/ablation_study.json --output-dir analysis/
```

## 📊 테스트 데이터셋

### 구성 (총 30개 질문)

| 카테고리 | 질문 수 | 설명 |
|---------|--------|------|
| concept | 10 | 개념 설명 (B+ tree, ACID 등) |
| comparison | 5 | 비교 분석 (clustered vs non-clustered) |
| procedure | 5 | 절차/알고리즘 (ARIES, 쿼리 최적화) |
| unit_specific | 5 | 특정 단원 질문 |
| out_of_scope | 5 | 범위 외 질문 (웹 검색 필요) |

### 단원 커버리지

- Unit 01-02: Relational Model, SQL
- Unit 03-06: Storage, Buffer, Compression
- Unit 07-10: Indexes, Hash Tables, Concurrency
- Unit 11-14: Sorting, Joins, Query Execution
- Unit 15-16: Query Optimization
- Unit 17-20: Concurrency Control (2PL, T/O, MVCC)
- Unit 21-22: Logging, Recovery
- Unit 23-24: Distributed Systems

## 📈 평가 지표

### Retrieval 평가

| 지표 | 설명 | 범위 |
|-----|------|------|
| Precision@K | 검색된 K개 중 관련 문서 비율 | 0-1 |
| Hit Rate | 관련 문서가 1개 이상 검색된 쿼리 비율 | 0-1 |
| MRR | 첫 번째 관련 문서의 역순위 평균 | 0-1 |

### Generation 평가 (LLM-as-Judge)

| 지표 | 설명 | 범위 |
|-----|------|------|
| Correctness | 정답과의 일치도 | 1-5 |
| Relevance | 질문에 대한 적절성 | 1-5 |
| Faithfulness | 검색된 문서에 근거 여부 | 1-5 |
| Completeness | 답변의 완성도 | 1-5 |

## 🔬 Ablation Study 설정

| 설정 | Rewrite | Web Search | Unit Filter |
|-----|---------|------------|-------------|
| Baseline | ❌ | ❌ | ❌ |
| +Rewrite | ✅ | ❌ | ❌ |
| +Web | ❌ | ✅ | ❌ |
| Full | ✅ | ✅ | ✅ |

## 📝 출력 파일

### 평가 결과 (JSON)
```json
{
  "config": {...},
  "retrieval": {
    "metrics": {"avg_precision_at_k": 0.72, ...},
    "details": [...]
  },
  "generation": {
    "metrics": {"avg_correctness": 3.9, ...},
    "details": [...]
  }
}
```

### 분석 출력
- `evaluation_report.md`: 마크다운 보고서
- `retrieval_comparison.png`: 검색 성능 차트
- `generation_comparison.png`: 생성 성능 차트
- `radar_comparison.png`: 레이더 차트
- `*.csv`: CSV 형식 결과

## ⚙️ 고급 옵션

### Judge 모델 변경
```bash
# GPT-4 사용
python evaluate_rag.py --judge-api openai --judge-model gpt-4o --ablation

# Claude 사용
python evaluate_rag.py --judge-api anthropic --judge-model claude-3-5-sonnet-20241022 --ablation
```

### 한국어 질문 사용
평가 스크립트는 기본적으로 한국어 질문(`question_ko`)을 사용합니다. 영어로 변경하려면 코드에서 `use_korean=False`로 설정하세요.

## 🔧 TA 시스템 연동

`evaluate_rag.py`는 다음 import를 시도합니다:
```python
from run_TAassistant import (
    retrieval_qa_pipline,
    build_quiz_chains,
    QueryRewriter,
    get_embeddings,
    load_model
)
```

TA 시스템 경로가 다른 경우, `_import_ta_system()` 메서드에서 `sys.path`를 수정하세요.

## 📌 주의사항

1. **API 비용**: LLM-as-Judge는 외부 API를 사용합니다. 30개 질문 평가 시 약 $0.5-1 (GPT-4o-mini 기준)
2. **GPU 필요**: TA 시스템 실행 시 CUDA GPU 권장
3. **시간 소요**: Full ablation study는 설정당 약 10-15분 소요

## 🐛 문제 해결

### "Failed to import TA system"
- `run_TAassistant.py`가 Python path에 있는지 확인
- 필요한 의존성이 모두 설치되었는지 확인

### "Judge evaluation failed"
- API 키가 올바르게 설정되었는지 확인
- 네트워크 연결 확인

### 메모리 부족
- `--device cpu` 옵션으로 CPU 모드 시도
- 배치 크기 줄이기
