# vector_db.py
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from file_loader import chunks  # 2단계에서 쪼갠 데이터 가져오기

# 1. 임베딩 모델 정의 (Ollama에 설치한 nomic 모델 사용)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. 로컬 벡터 DB 생성 및 저장
# './chroma_db'라는 로컬 폴더에 데이터가 물리적으로 저장됩니다.
db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("로컬 벡터 데이터베이스 구축 완료!")

# 3. 간단한 검색 테스트
query = "건강하게 다이어트하는 방법을 알려줘"
# 질문과 가장 유사한 문서 조각 2개 찾기
docs = db.similarity_search(query, k=2) 

print("\n[검색된 관련 문서 내용]")
for i, doc in enumerate(docs):
    print(f"조각 {i+1}: {doc.page_content}\n")