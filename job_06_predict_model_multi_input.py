import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from keras.utils import to_categorical
from keras.layers import Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from job_03_Preprocessing import subtract_one_text, preprocess_text, ToTokenizer
from keras.models import load_model
from job_04_customer_layer import TransformerEncoder

raw_data = pd.read_csv('./crawling_data/naver_predict_webtoons.csv')

df = raw_data.dropna(axis=0)

with open('./models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)

title_max_size, plot_max_size = 20, 327
title_word_size, plot_word_size = 10279, 33137

with open(f'./models/title_token_max_{title_max_size}.pickle', 'rb') as f:
    token = pickle.load(f)
    title_token = ToTokenizer(token)
with open(f'./models/plot_token_max_{plot_max_size}.pickle', 'rb') as f:
    token = pickle.load(f)
    plot_token = ToTokenizer(token)

label = encoder.classes_

print(label)

def extract_category(text):
    for category in label:
        if category in text:
            return category
    return '기타'

df.category = df.category.apply(extract_category)

df = df[df.category != '기타']
print(df.category.value_counts())

X = df[['title', 'plot']]
Y = df['category']

print(X.shape, Y.shape)

labeled_y = encoder.transform(Y)
onehot_y = to_categorical(labeled_y)
print(onehot_y)

title = X['title'].apply(preprocess_text)
plot = X['plot'].apply(preprocess_text)

pre_X = pd.concat([title, plot], axis=1)

pre_X['title'] = pre_X['title'].apply(subtract_one_text)
pre_X['plot'] = pre_X['plot'].apply(subtract_one_text)

title_x = title_token.tokenizer_text(pre_X['title'])
plot_x = plot_token.tokenizer_text(pre_X['plot'])

print(title_x[:5])
print(plot_x[:5])

title_pad = title_token.padding_text()
plot_pad = plot_token.padding_text()

model = load_model('transformer_multifeature.h5', custom_objects={'TransformerEncoder': TransformerEncoder})
print(model.inputs)

inputs = [title_pad, plot_pad]

preds = model.predict(inputs)
print(preds)

predict_section = []
for pred in preds :
    most = label[np.argmax(pred)]
    pred[np.argmax([pred])] = 0
    second = label[np.argmax(pred)]
    predict_section.append([most, second])
    # predict_section.append(label[np.argmax(pred)])
print(predict_section)

df['predict'] = predict_section
df['category'] = Y
df = df[['title', 'category', 'predict']]
print(df.head())

score = model.evaluate(inputs, onehot_y)
print(score[1])

# 모델 예측 (예: y_pred는 모델의 예측 결과, y_test는 실제 값)
y_pred = model.predict(inputs)

# 예측 결과가 확률로 나오는 경우, 가장 높은 확률을 가진 인덱스를 선택
y_pred_classes = np.argmax(y_pred, axis=1)  # 가장 큰 값의 인덱스 (클래스 예측)
y_true_classes = np.argmax(onehot_y, axis=1)  # 실제 클래스

# F1-Score 계산
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')  # 평균 방법을 weighted로 설정 (각 클래스의 샘플 수에 비례)
print("F1-Score:", f1)

# 추가로 classification_report 출력 (정밀도, 재현율, F1-score 등)
print(classification_report(y_true_classes, y_pred_classes))

# df['ox'] = 0
# for i in range(len(df)):
#     if df.loc[i, 'category'] in df.loc[i, 'predict']:
#         df.loc[i, 'ox'] = 1
#
# print(df.ox.mean())
