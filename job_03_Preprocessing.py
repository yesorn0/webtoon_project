import pickle
import pandas as pd
import numpy as np
from numpy.ma.core import ones
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt, Komoran
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os

def preprocess_text(text):
    okt = Okt()
    text = re.sub('[^가-힣]', ' ', text)            # 한글만 남기고 제거
    morphs = okt.morphs(text, stem=True)                        # 형태소 분석 + 원형 복원
    return morphs

def subtract_one_text(text) :
    words = []
    for word in text:
        if len(word) > 1 :
            words.append(word)
    return ' '.join(words)

class ToTokenizer:
    def __init__(self, istoken = None):
        if istoken is None : self.token = Tokenizer()
        else : self.token = istoken
        self.word_size = None
        self.token_text = None
        self.max = 0

    def tokenizer_text(self, text):
        self.word_size = len(self.token.word_index) + 1
        self.token_text = self.token.texts_to_sequences(text)
        return self.token_text
    def max_token_text(self):
        if self.token_text is None :
            print('Tokend text is Empty, execute tokenizer_text')
            return
        for sentence in self.token_text:
            if self.max < len(sentence):
                self.max = len(sentence)
        return self.max

    def padding_text(self):
        if self.max == 0 : self.max_token_text()
        x_pad = pad_sequences(self.token_text, self.max)
        return x_pad

    def save_tokenizer(self, name):
        with open('./models/{}_token_max_{}.pickle'.format(name, self.max), 'wb') as f:
            pickle.dump(self.token, f)

# test_title = preprocess_text(X.at[0, 'title'])
# test_plot = preprocess_text(X.at[0, 'plot'])
# print(test_title, test_plot)

if __name__ == '__main__' :
    flag = 1
    os.makedirs('./models', exist_ok=True)
    raw_data = pd.read_csv('./crawling_data/webtoons.csv')
    df = raw_data.dropna(axis=0)
    X = df[['title', 'plot']]
    Y = df['category']

    print(X.shape, Y.shape)

    encoder = LabelEncoder()
    labeled_y = encoder.fit_transform(Y)
    print(Y[:5])
    print(labeled_y[:5])
    label = encoder.classes_
    print(label)
    with open('./models/encoder.pickle', 'wb') as f:
        pickle.dump(encoder, f)

    onehot_y = to_categorical(labeled_y)
    print(onehot_y)

    okt = Okt()
    if flag == 1:
        titles = X['title'].apply(preprocess_text)
        plots = X['plot'].apply(preprocess_text)

        df = pd.concat([titles, plots], axis=1)

        df['title'] = df['title'].apply(subtract_one_text)
        df['plot'] = df['plot'].apply(subtract_one_text)

        print(df.head())
        df.info()

        title_token = ToTokenizer()
        plot_token = ToTokenizer()

        title_token.token.fit_on_texts(df['title'])
        plot_token.token.fit_on_texts(df['plot'])

        title_token.tokenizer_text(df['title'])
        plot_token.tokenizer_text(df['plot'])

        title_pad = title_token.padding_text()
        plot_pad = plot_token.padding_text()

        X_pad = np.concatenate((title_pad, plot_pad), axis=1)
        print(X_pad)

        title_token.save_tokenizer('title')
        plot_token.save_tokenizer('plot')

        X_train, X_test, y_train, y_test = train_test_split(
            X_pad, onehot_y, test_size=0.1)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)

        title_ws = title_token.word_size
        plot_ws = plot_token.word_size

        np.save(f'./crawling_data/X_train_wordsize_{title_ws}_{plot_ws}.npy', X_train)
        np.save(f'./crawling_data/X_test_wordsize_{title_ws}_{plot_ws}.npy', X_test)
        np.save(f'./crawling_data/y_train_wordsize_{title_ws}_{plot_ws}.npy', y_train)
        np.save(f'./crawling_data/y_test_wordsize_{title_ws}_{plot_ws}.npy', y_test)

    if flag == 2:
        texts = X['plot'].apply(preprocess_text)
        texts = texts.apply(subtract_one_text)
        token = ToTokenizer()
        token.token.fit_on_texts(X['plot'])
        token.tokenizer_text(X['plot'])
        X_pad = token.padding_text()
        token.save_tokenizer('platform')
        word_size = token.word_size
        print(word_size)

        np.save(f'./crawling_data/total_X_wordsize_{word_size}.npy', X_pad)
        np.save(f'./crawling_data/total_y_wordsize_{word_size}.npy', onehot_y)

        kakao_data = pd.read_csv('./crawling_data/kakao_data.csv')
        naver_data = pd.read_csv('./crawling_data/naver_data.csv')
        lezhin_data = pd.read_csv('./crawling_data/lezhin_data.csv')

        kakao = token.tokenizer_text(kakao_data['plot'])
        kakao_pad = token.padding_text()
        naver = token.tokenizer_text(naver_data['plot'])
        naver_pad = token.padding_text()
        lezhin = token.tokenizer_text(lezhin_data['plot'])
        lezhin_pad = token.padding_text()

        kakao_y = encoder.fit_transform(kakao_data['category'])
        kakao_y = to_categorical(kakao_y)
        naver_y = encoder.fit_transform(naver_data['category'])
        naver_y = to_categorical(naver_y)
        lezhin_y = encoder.fit_transform(lezhin_data['category'])
        lezhin_y = to_categorical(lezhin_y)

        print(naver_pad[:5])
        print(naver_y[:5])
        print(kakao_pad[:5])
        print(kakao_y[:5])
        print(lezhin_pad[:5])
        print(lezhin_y[:5])

        np.save(f'./crawling_data/kakao_X_wordsize_{word_size}.npy', kakao_pad)
        np.save(f'./crawling_data/kakao_y_wordsize_{word_size}.npy', kakao_y)
        np.save(f'./crawling_data/naver_X_wordsize_{word_size}.npy', naver_pad)
        np.save(f'./crawling_data/naver_y_wordsize_{word_size}.npy', naver_y)
        np.save(f'./crawling_data/lezhin_X_wordsize_{word_size}.npy', lezhin_pad)
        np.save(f'./crawling_data/lezhin_y_wordsize_{word_size}.npy', lezhin_y)