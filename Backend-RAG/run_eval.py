import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


# RAG 시스템 연동을 위한 라이브러리 (사용하는 프레임워크에 맞게 수정 가능)
# RAG 시스템 import
from run_TAassistant import retrieval_qa_pipline

# ==========================================
# 1. 환경 설정 및 데이터 준비
# ==========================================

# OpenAI API 키 설정 (평가 모델인 GPT-4가 사용)
os.environ["OPENAI_API_KEY"] = ""  # <-- 여기에 실제 API Key 입력

# 평가용 데이터셋 파일명 (앞서 만든 JSON 파일)
TESTSET_FILE = 'testset_lec01_3.json' 


qa_chain, rewriter = retrieval_qa_pipline(
    device_type="cuda",  # 또는 "cpu"
    use_history=False,
    promptTemplate_type="qwen2.5",
    mode="answer",
)

# ==========================================
# 2. RAG 시스템 연결 함수 (★여기가 핵심★)
# ==========================================
def generate_rag_response(question):
    """
    여러분의 RAG 시스템에 질문을 던지고, 답변과 검색된 문서를 반환하는 함수입니다.
    실제 RAG 코드를 이 안에 넣으셔야 합니다.
    """
    
    # --- [TODO: 실제 RAG 연결 파트] -----------------------
    # 예시: LangChain을 쓴다면?
    # result = my_qa_chain({"query": question})
    # answer_text = result['result']
    # retrieved_docs = [doc.page_content for doc in result['source_documents']]
    # ----------------------------------------------------
        # 쿼리 리라이트 (선택)
    if rewriter:
        question = rewriter.rewrite(question)

    # RAG 체인 호출
    result = qa_chain.invoke({"query": question})
    
    # 답변 추출
    answer_text = result['result']
    
    # 검색된 문서 내용 추출
    retrieved_docs = [doc.page_content for doc in result['source_documents']]

    # --- [TEST: 현재는 테스트를 위해 더미(Dummy) 응답 반환] ---
    # 코드를 돌려보시려면 아래 주석을 풀고 실제 로직을 넣으세요.
    # 지금은 평가 코드가 잘 도는지 확인하기 위해 가짜 데이터를 줍니다.
    
    # answer_text = "관계형 모델은 1970년 Ted Codd가 제안했습니다. [Lec 01, Slide 5]"
    
    # # 실제 검색된 문서 내용이라 가정
    # retrieved_docs = [
    #     "The relational model was proposed by Ted Codd in 1970 to protect users from details of storage.",
    #     "Data independence is a key feature of the relational model."
    # ]
    
    return answer_text, retrieved_docs

# ==========================================
# 3. 메인 실행 로직
# ==========================================
def main():
    # 1) 골드 데이터셋 로드
    if not os.path.exists(TESTSET_FILE):
        print(f"오류: {TESTSET_FILE} 파일이 없습니다. 먼저 데이터셋을 생성해주세요.")
        return

    with open(TESTSET_FILE, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)

    print(f"▶ 총 {len(gold_data)}개의 평가 데이터를 로드했습니다.")
    
    # 2) RAG 답변 생성 (Inference Loop)
    evaluation_data = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': []
    }

    print("▶ RAG 시스템 답변 생성 중...", end="")
    for idx, item in enumerate(gold_data):
        q = item['question']
        gt = item['ground_truth']
        
        # RAG 함수 호출
        try:
            ans, ctxs = generate_rag_response(q)
        except Exception as e:
            print(f"\n[Error] 질문 '{q}' 처리 중 오류 발생: {e}")
            ans = "Error generating response"
            ctxs = []

        # 데이터 수집
        evaluation_data['question'].append(q)
        evaluation_data['answer'].append(ans)
        evaluation_data['contexts'].append(ctxs) # 리스트 형태 그대로 저장
        evaluation_data['ground_truth'].append(gt)
        
        if idx % 5 == 0: print(".", end="") # 진행상황 표시

    print("\n▶ 답변 생성 완료! Ragas 평가를 시작합니다 (시간이 조금 걸립니다)...")

    # 3) Ragas 평가 실행
    # DataFrame으로 변환 후 HuggingFace Dataset으로 변환
    df = pd.DataFrame(evaluation_data)
    dataset = Dataset.from_pandas(df)

    # 평가 지표 설정 (기획서 요구사항 반영)
    # faithfulness: 환각 여부 (답변이 문맥에 근거하는가)
    # answer_relevancy: 답변 적절성 (질문에 맞는 답인가)
    # context_recall: 검색 재현율 (정답을 위한 문서가 검색되었는가)
    # context_precision: 검색 정확도 (불필요한 문서는 없는가)
    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision
    ]

    results = evaluate(dataset=dataset, metrics=metrics)

    # ==========================================
    # 4. 결과 저장 및 출력
    # ==========================================
    
    # 결과 DataFrame 변환
    result_df = results.to_pandas()
    
    # 파일로 저장 (타임스탬프 포함 가능)
    output_filename = "rag_evaluation_results.csv"
    result_df.to_csv(output_filename, index=False, encoding='utf-8-sig') # 엑셀 깨짐 방지 utf-8-sig

    print("\n" + "="*50)
    print(f"✅ 평가 완료! 결과가 '{output_filename}'에 저장되었습니다.")
    print("="*50)
    print(results)
    
    # 간단한 요약 출력
    print("\n[평가 요약]")
    print(f"- Faithfulness: {results['faithfulness']:.4f}")
    print(f"- Answer Relevancy: {results['answer_relevancy']:.4f}")
    print(f"- Context Recall: {results['context_recall']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()