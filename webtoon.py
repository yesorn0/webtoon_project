# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow
# from PyQt5 import QtGui, uic
# from PyQt5.QtGui import QFont
#
# class MainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         uic.loadUi("webtoon.ui", self)  # UI 불러오기
#
#         # 글꼴 설정
#         font = QFont("Baskerville Old Face", 15)
#         self.text.setFont(font)
#         self.message.setFont(font)
#
#         #워터마크(placeholder) 텍스트 설정
#         self.text.setPlaceholderText("원하는 웹툰의 장르를 입력하세요")
#
#         # 버튼 클릭 시 동작 연결
#         self.send.clicked.connect(self.send_message)
#
#     def send_message(self):
#         message = self.text.text()
#         if message:
#             self.message.append(f"나: {message}")
#             self.text.clear()
#             self.message.moveCursor(QtGui.QTextCursor.End)
#
#             # AI의 응답
#             response = self.get_ai_response(message)
#             self.message.append(f"AI: {response}")
#             self.message.moveCursor(QtGui.QTextCursor.End)
#
#     def get_ai_response(self, message):
#         message = message.lower()
#         if "로맨스" in message:
#             return "'그녀의 사생활', '연애혁명'"
#         elif "드라마" in message:
#             return "'미생', '치즈인더트랩'"
#         elif "판타지" in message:
#             return "'나 혼자만 레벨업', '신의 탑'"
#         elif "액션" in message:
#             return "'고수', '열렙전사'"
#         else:
#             return "죄송해요. 해당 장르에 대한 추천이 없습니다. 로맨스, 드라마, 판타지, 액션 중에서 입력해 주세요."
#
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())

import sys
import random

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtGui, uic
from PyQt5.QtGui import QFont
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import pandas as pd
import re

# 사용자 입력에서 장르 키워드만 추출하는 함수
def extract_genre_keyword(text):
    genres = ['로맨스', '드라마', '판타지', '액션', '무협', '스릴러', '개그', '감성']  # 필요 시 확장
    for genre in genres:
        if genre in text:
            return genre
    return None

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("webtoon.ui", self)  # UI 파일 불러오기

        # 글꼴 설정
        font = QFont("Baskerville Old Face", 15)
        self.text.setFont(font)
        self.message.setFont(font)

        # 워터마크 텍스트 설정
        self.text.setPlaceholderText("원하는 웹툰의 장르를 입력하세요 (예: 판타지 추천해줘)")

        # 버튼 클릭 이벤트 연결
        self.send.clicked.connect(self.send_message)

        # 모델 및 전처리 도구 로드
        self.model = load_model('./models/00_title_plot_lstm_66.h5')
        self.encoder = pickle.load(open('./models/encoder_webtoon.pickle', 'rb'))
        self.plot_token = pickle.load(open('./models/plot_token_max_327.pickle', 'rb'))
        self.max_size = 327
        self.label = self.encoder.classes_

        # 여러 웹툰 데이터를 병합해서 사용
        df1 = pd.read_csv('./crawling_data/naver_category_plot.csv')
        df2 = pd.read_csv('./crawling_data/kakao_category_plot.csv')
        df3 = pd.read_csv('./crawling_data/lezhin_category_plot.csv')
        self.webtoon_data = pd.concat([df1, df2, df3], ignore_index=True)
        self.webtoon_data.dropna(subset=["category", "title"], inplace=True)

    def send_message(self):
        message = self.text.text()
        if message:
            self.message.append(f"나: {message}")
            self.text.clear()
            self.message.moveCursor(QtGui.QTextCursor.End)

            # AI의 응답
            response = self.get_ai_response(message)
            self.message.append(f"AI: {response}")
            self.message.moveCursor(QtGui.QTextCursor.End)

    def get_ai_response(self, message):
        genre = extract_genre_keyword(message)
        if genre is None:
            return "죄송해요. 장르를 인식하지 못했어요. 예: 로맨스, 드라마, 판타지 등 입력해 주세요."

        return self.get_webtoon_recommendations(genre)

    def get_webtoon_recommendations(self, category):
        filtered = self.webtoon_data[self.webtoon_data['category'] == category]
        if filtered.empty:
            return f"죄송해요. '{category}' 장르의 웹툰을 찾을 수 없어요."
        rand = random.randint(10,len(filtered))
        recommendations = ''.join(filtered['title'].dropna().unique()[rand])
        return f"내가 추천 하는 {category} 웹툰은 '{recommendations}' 이야"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
