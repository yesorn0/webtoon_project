import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping

title_max_len, plot_max_len = 20, 327
title_word_size, plot_word_size = 10279, 33137

X_train = np.load(f'./crawling_data/X_train_wordsize_{title_word_size}_{plot_word_size}.npy', allow_pickle=True)
X_test = np.load(f'./crawling_data/X_test_wordsize_{title_word_size}_{plot_word_size}.npy', allow_pickle=True)
y_train = np.load(f'./crawling_data/y_train_wordsize_{title_word_size}_{plot_word_size}.npy', allow_pickle=True)
y_test = np.load(f'./crawling_data/y_test_wordsize_{title_word_size}_{plot_word_size}.npy', allow_pickle=True)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

title_input = X_train[:, :title_max_len]
plot_input = X_train[:, title_max_len:title_max_len + plot_max_len]

title_test = X_test[:, :title_max_len]
plot_test = X_test[:, title_max_len:title_max_len+plot_max_len]

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
checkpoint = ModelCheckpoint('./models/best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
# 타이틀 모델 불러오기
title_model = load_model('./models/title_lstm.h5')

# 새 모델 구성
plot_model = Sequential()
plot_model.add(Embedding(plot_word_size, 256))
plot_model.build(input_shape=(None, plot_max_len))
plot_model.add(Conv1D(32, kernel_size=8, padding='same', activation='relu'))
plot_model.add(MaxPool1D(pool_size = 4))
plot_model.add(LSTM(128, activation='tanh', return_sequences=True))
plot_model.add(Dropout(0.3))
plot_model.add(LSTM(64, activation='tanh', return_sequences=True))
plot_model.add(Dropout(0.3))
plot_model.add(LSTM(32, activation='tanh', return_sequences=True))
plot_model.add(Dropout(0.3))
plot_model.add(LSTM(64, activation='tanh', return_sequences=True))
plot_model.add(Dropout(0.3))
plot_model.add(LSTM(128, activation='tanh'))
plot_model.add(Dense(128, activation='relu'))
plot_model.add(Dense(64, activation='relu'))
plot_model.add(Dense(y_train.shape[1], activation='softmax'))

# title 모델의 LSTM 가중치만 가져와서 plot 모델에 이식
for i in range(1, 8):  # title_model의 Conv1D~LSTM 블록
    plot_model.layers[i].set_weights(title_model.layers[i].get_weights())

plot_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
plot_model.summary()

# plot 데이터 학습
fit_hist = plot_model.fit(plot_input, y_train, epochs=50, batch_size=64,
                          validation_data=(plot_test, y_test), callbacks=[early_stop, checkpoint])

score = plot_model.evaluate(plot_test, y_test, verbose=0)
print('Final test set accuracy', score[1])

plt.plot(fit_hist.history['val_accuracy'], label='val accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.savefig('./models/transfer_model_{}_accuracy.jpg'.format(score[1]))
plt.show()
