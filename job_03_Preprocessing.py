import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt, Komoran
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import os

os.makedirs('./models', exist_ok=True)
raw_data = pd.read_csv('./crawling_data/webtoon_category_plot.csv')
raw_data.info()
print(raw_data.head())
print(raw_data.category.value_counts())

print(raw_data.columns)

X = raw_data[['title', 'plot']]
Y = raw_data['category']

print(X.shape, Y.shape)

okt = Okt()

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)
print(labeled_y[:5])
label = encoder.classes_
print(label)

with open('./models/encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

onehot_y = to_categorical(labeled_y)
print(onehot_y)

def preprocess_text(text):
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
    def __init__(self):
        self.token = Tokenizer()
        self.word_size = None
        self.token_text = None
        self.max = 0

    def tokenizer_text(self, text):
        self.token.fit_on_texts(text)
        self.token_text = self.token.texts_to_sequences(text)
        self.word_size = len(self.token.word_index) + 1

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

titles = X['title'].apply(preprocess_text)
plots = X['plot'].apply(preprocess_text)

df = pd.concat([titles, plots], axis=1)

df['title'] = df['title'].apply(subtract_one_text)
df['plot'] = df['plot'].apply(subtract_one_text)

print(df.head())
df.info()

title_token = ToTokenizer()
plot_token = ToTokenizer()

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