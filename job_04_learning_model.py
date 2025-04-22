import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint

X_train = np.load('./crawling_data/X_train_wordsize_10122_32541.npy', allow_pickle=True)
X_test = np.load('./crawling_data/X_test_wordsize_10122_32541.npy', allow_pickle=True)
y_train = np.load('./crawling_data/y_train_wordsize_10122_32541.npy', allow_pickle=True)
y_test = np.load('./crawling_data/y_test_wordsize_10122_32541.npy', allow_pickle=True)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

title_max_len, plot_max_len = 16, 327

with open('./models/title_token_max_{}.pickle'.format(title_max_len), 'rb') as f:
    title_token = pickle.load(f)
with open('./models/plot_token_max_{}.pickle'.format(plot_max_len), 'rb') as f:
    plot_token = pickle.load(f)

title_word_size = title_token.word_size
plot_word_size = plot_token.word_size

title_input = X_train[:, :title_max_len]
plot_input = X_train[:, title_max_len:title_max_len + plot_max_len]

title_model = Sequential()
title_model.add(Embedding(title_word_size, 128, input_length=title_max_len))
title_model.add(LSTM(128))
title_model.add(Dense(y_train.shape[1], activation='softmax'))

title_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

title_model.summary()

# 저장 경로 설정
checkpoint_path = './models/title_lstm.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)

# 훈련
title_model.fit(title_input, y_train, epochs=10, batch_size=64, validation_split=0.2, callbacks=[checkpoint])

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# 타이틀 모델 불러오기
pretrained_model = load_model('./models/title_lstm.h5')

# 새 모델 구성
plot_model = Sequential()
plot_model.add(Embedding(plot_word_size, 128, input_length=plot_max_len))
plot_model.add(LSTM(128))
plot_model.add(Dense(y_train.shape[1], activation='softmax'))

# title 모델의 LSTM 가중치만 가져와서 plot 모델에 이식
plot_model.layers[1].set_weights(pretrained_model.layers[1].get_weights())
plot_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
plot_model.summary()

# plot 데이터 학습
fit_hist = plot_model.fit(plot_input, y_train, epochs=20, batch_size=64, validation_split=0.2)
# fit_hist = model.fit(X_train, y_train, batch_size=128,
#                      epochs=10, validation_data=(X_test, y_test))

score = plot_model.evaluate(X_test, y_test, verbose=0)
print('Final test set accuracy', score[1])
plot_model.save('./models/news_section_classfication_model_{}.h5'.format(score[1]))

plt.plot(fit_hist.history['val_accuracy'], label='val accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.savefig('./models/news_section_classfication_model_{}_accuracy.jpg'.format(score[1]))
plt.show()
