import numpy as np
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train = np.load('./crawling_data/X_train_wordsize_2544_10312.npy', allow_pickle=True)
X_test = np.load('./crawling_data/X_test_wordsize_2544_10312.npy', allow_pickle=True)
y_train = np.load('./crawling_data/y_train_wordsize_2544_10312.npy', allow_pickle=True)
y_test = np.load('./crawling_data/y_test_wordsize_2544_10312.npy', allow_pickle=True)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# X_train과 X_test에서 첫 번째 샘플을 출력
print("First sample in X_train:", X_train[0])
print("First sample in X_test:", X_test[0])

# 샘플 길이가 맞는지 확인
print("Length of first sample in X_train:", len(X_train[0]))
print("Length of first sample in X_test:", len(X_test[0]))

model = Sequential()
model.add(Embedding(12856, 300))
model.build(input_shape=(None, 164))
model.add(Conv1D(64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size = 1))
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
fit_hist = model.fit(X_train, y_train, batch_size=128,
                     epochs=10, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Final test set accuracy', score[1])
model.save('./models/news_section_classfication_model_{}.h5'.format(score[1]))

plt.plot(fit_hist.history['val_accuracy'], label='val accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.savefig('./models/news_section_classfication_model_{}_accuracy.jpg'.format(score[1]))
plt.show()
