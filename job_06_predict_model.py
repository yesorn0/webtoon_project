import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from job_03_Preprocessing import subtract_one_text, preprocess_text
from keras.models import load_model

raw_data = pd.read_csv('./crawling_data/naver_predict_webtoons.csv')

df = raw_data.dropna(axis=0)

with open('./models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)

max_size = 327

with open(f'./models/plot_token_max_{max_size}.pickle', 'rb') as f:
    plot_token = pickle.load(f)

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

X = df['plot']
Y = df['category']

print(X.shape, Y.shape)

labeled_y = encoder.transform(Y)
onehot_y = to_categorical(labeled_y)
print(onehot_y)

# X.title = X.title.apply(preprocess_text)
X = X.apply(preprocess_text)

# X.title = X.title.apply(subtract_one_text)
X = X.apply(subtract_one_text)

tokened_x = plot_token.texts_to_sequences(X)
print(tokened_x)

for i in range(len(tokened_x)):
    if len(tokened_x[i]) > max_size :
        tokened_x[i] = tokened_x[i][:max_size]
x_pad = pad_sequences(tokened_x, max_size)
print(x_pad)

model = load_model('./models/transformer_multifeature.h5')

preds = model.predict(x_pad)
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

score = model.evaluate(x_pad, onehot_y)
print(score[1])

# 모델 예측 (예: y_pred는 모델의 예측 결과, y_test는 실제 값)
y_pred = model.predict(x_pad)

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
