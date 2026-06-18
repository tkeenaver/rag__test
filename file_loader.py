# file_loader.py
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 문서 불러오기
# 실습용 company_rules.txt 파일이 같은 폴더에 있어야 합니다.
loader = TextLoader("rag_documents.txt", encoding="utf-8")
documents = loader.load()

# 2. 문서 쪼개기 (Text Splitting)
# 4GB VRAM 환경이므로 LLM이 한 번에 읽을 문맥을 400자 내외로 쪼갭니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,       # 쪼갤 글자 수
    chunk_overlap=50      # 문맥 연결을 위해 겹칠 글자 수
)
chunks = text_splitter.split_documents(documents)

print(f"문서가 총 {len(chunks)}개의 조각으로 쪼개졌습니다.")