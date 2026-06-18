# rag_chain.py
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_rag_response(user_question):
    # 1. 기존에 저장된 벡터 DB 불러오기
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})

    # 2. 로컬 LLM 설정 (Qwen 2.5 3B)
    llm = ChatOllama(model="qwen2.5:3b", temperature=0.2)

    # 3. 프롬프트 템플릿 설계
    template = """당신은 KoAlpava 비서입니다. 
아래 제공된 문서만을 바탕으로 질문에 답하세요. 
만약 문서 내용으로 답변을 알 수 없다면, 억지로 지어내지 말고 "해당 내용은 문서에서 찾을 수 없습니다."라고 답변하세요.

[KoAlpaca 문서]
{context}

질문: {question}
답변:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 4. 랭체인 파이프라인(Chain) 구성 (LCEL 문법)
    # 파이썬의 비트 연산자(|)를 오버로딩하여 데이터가 물 흐르듯 전달되게 만든 랭체인 특유의 문법입니다.
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. 실행 후 결과 반환
    return rag_chain.invoke(user_question)

# ==========================================
# 추가된 부분: 모듈 단독 실행 및 터미널 테스트용 코드
# ==========================================
if __name__ == "__main__":
    print("🤖 로컬 RAG 시스템 테스트 창입니다. (종료하려면 'q' 입력)")
    print("-" * 50)
    
    while True:
        query = input("\n질문을 입력하세요: ")
        if query.lower() == 'q':
            print("테스트를 종료합니다.")
            break
            
        if not query.strip():
            continue
            
        print("데이터 검색 및 답변 생성 중...")
        # 위에서 정의한 함수를 호출하여 RAG 결과값 받아오기
        answer = get_rag_response(query)
        print(f"\n>> AI 답변: {answer}")