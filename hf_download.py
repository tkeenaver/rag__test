from datasets import load_dataset
from huggingface_hub import login # 로그인 함수 추가

# 1. 방금 복사한 내 토큰을 입력하여 인증하기 (경고 메시지 해결!)
login("hf_xxxx") # 이자리에 key 입력

print("허깅페이스에서 데이터를 다운로드합니다...")

# 2. 네임스페이스가 포함된 데이터셋 불러오기
dataset = load_dataset("beomi/KoAlpaca-v1.1a")

# RAG 문서 DB에 넣을 텍스트 파일(context)과 
# 수강생 테스트용 질문 파일(question)을 분리해서 저장합니다.
with open("rag_documents.txt", "w", encoding="utf-8") as doc_file, \
     open("test_questions.txt", "w", encoding="utf-8") as q_file:
    
    # 실습을 위해 50개만 추출
    for i in range(50):
        data = dataset["train"][i]
        
        doc_file.write(f"--- [위키 지식 {i+1}] ---\n")
        doc_file.write(data['output'].strip() + "\n\n")
        
        q_file.write(f"질문 {i+1}: {data['instruction'].strip()}\n")
        q_file.write(f"  -> 예상 답변(요약): {data['output'][:40]}...\n\n")

print("다운로드 및 텍스트 파일 변환이 완료되었습니다!")