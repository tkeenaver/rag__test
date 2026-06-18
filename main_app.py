# main_app.py
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QTextEdit, QLabel
from rag_chain import get_rag_response

class RagChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        # 레이아웃 설정
        layout = QVBoxLayout()
        
        # UI 요소 생성
        self.label = QLabel("KoAlpaca AI 비서 (Local RAG)", self)
        self.txt_input = QLineEdit(self)
        self.txt_input.setPlaceholderText("궁금한 점을 입력하세요... (예: 자본주의의 장점과 단점이 뭐야?)")
        
        self.btn_submit = QPushButton("AI에게 질문하기", self)
        # 버튼을 누르면 내부 함수(send_question)가 실행되도록 슬롯 연결
        self.btn_submit.clicked.connect(self.send_question)
        
        self.txt_output = QTextEdit(self)
        self.txt_output.setReadOnly(True)  # 답변 창은 읽기 전용으로
        
        # 레이아웃에 요소 배치
        layout.addWidget(self.label)
        layout.addWidget(self.txt_input)
        layout.addWidget(self.btn_submit)
        layout.addWidget(self.txt_output)
        
        self.setLayout(layout)
        self.setWindowTitle("프라이빗 기업용 RAG 시스템 데모")
        self.resize(500, 400)

    def send_question(self):
        question = self.txt_input.text()
        if not question:
            return
            
        self.txt_output.append(f"👨‍💻 질문: {question}")
        self.txt_input.clear()
        
        # 4단계에서 만든 로컬 RAG 파이프라인 호출
        # 연산 시 4GB VRAM을 활용하므로 1~3초 내외로 깔끔하게 답변이 생성됩니다.
        response = get_rag_response(question)
        
        self.txt_output.append(f"🤖 AI 담당자: {response}")
        self.txt_output.append("-" * 40)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = RagChatApp()
    ex.show()
    sys.exit(app.exec())