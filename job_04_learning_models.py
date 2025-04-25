import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, EarlyStopping

class Model:
    def __init__(self, shapes, word_size):
        self.model = None
        self.input_shape = shapes[0]
        self.output_shape = shapes[1]
        self.word_size = word_size
        self.history = None
        self.score = None
        self.checkpoint_path = './models/checkpoint.h5'
        self.checkpoint = ModelCheckpoint(self.checkpoint_path, monitor='val_accuracy', save_best_only=True)
        self.early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    def set_model(self, dimensions):
        model = Sequential()
        model.add(Embedding(self.word_size, dimensions))
        model.build(input_shape=(None, self.input_shape))
        model.add(Conv1D(32, kernel_size=16, padding='same', activation='relu'))
        model.add(MaxPool1D(pool_size=8))
        model.add(LSTM(128, activation='tanh', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(64, activation='tanh', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(32, activation='tanh'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.output_shape, activation='softmax'))
        self.model = model

    def fine_tuning(self, x, y, validation_data, epochs=5, batch_size=32, freeze_embedding=True):
        self.model.load_weights(self.checkpoint_path)
        print(">> Loaded weights for fine-tuning.")

        if freeze_embedding:
            for layer in self.model.layers[:3]:  # Embedding - LSTM 첫번째 까지 Freeze
                layer.trainable = False
            print(">> Embedding & Conv1D layers frozen.")

        self.history = self.model.fit(
            x, y,
            validation_data=validation_data,
            epochs=epochs, batch_size=batch_size,
            callbacks=[self.checkpoint, self.early_stop])

    def compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def summary(self):
        self.model.summary()

    def save_model(self, name):
        self.model.save(f'./models/{name}.h5')

    def fit(self, x, y, validation_data, epochs = 5, batch_size = 32):
        self.history = self.model.fit(x, y, epochs = epochs, batch_size = batch_size,
                             validation_data=validation_data, callbacks=[self.checkpoint, self.early_stop])

    def evaluate(self, x, y):
        self.score = self.model.evaluate(x, y, verbose = 1)
        print('Final test set accuracy', self.score[1])

    def show_plot(self):
        plt.plot(self.history.history['val_accuracy'], label='val accuracy')
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.legend()
        plt.savefig('./models/model_{}_accuracy.jpg'.format(self.score[1]))
        plt.show()

if __name__ == '__main__':
    words_max, words_size = 671, 183067

    naver_X = np.load(f'./crawling_data/naver_X_wordsize_{words_size}.npy', allow_pickle=True)
    naver_y = np.load(f'./crawling_data/naver_y_wordsize_{words_size}.npy', allow_pickle=True)
    kakao_X = np.load(f'./crawling_data/kakao_X_wordsize_{words_size}.npy', allow_pickle=True)
    kakao_y = np.load(f'./crawling_data/kakao_y_wordsize_{words_size}.npy', allow_pickle=True)
    lezhin_X = np.load(f'./crawling_data/lezhin_X_wordsize_{words_size}.npy', allow_pickle=True)
    lezhin_y = np.load(f'./crawling_data/lezhin_y_wordsize_{words_size}.npy', allow_pickle=True)

    # --- Pre-train with NAVER ---
    train_X, test_X, train_y, test_y = train_test_split(naver_X, naver_y, test_size=0.1)

    model = Model([words_max, train_y.shape[1]], words_size)
    model.set_model(128)
    model.compile()
    model.summary()
    model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=3)
    model.save_model('naver')

    # --- Fine-tune with KAKAO ---
    train_X, test_X, train_y, test_y = train_test_split(kakao_X, kakao_y, test_size=0.1)
    model.fine_tuning(train_X, train_y, validation_data=(test_X, test_y), epochs=3)
    model.save_model('kakao')

    # --- Fine-tune with LEZHIN ---
    train_X, test_X, train_y, test_y = train_test_split(lezhin_X, lezhin_y, test_size=0.1)
    model.fine_tuning(train_X, train_y, validation_data=(test_X, test_y), epochs=3)
    model.save_model('lezhin')

    X = np.concatenate([naver_X, kakao_X, lezhin_X])
    y = np.concatenate([naver_y, kakao_y, lezhin_y])

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=42)

    model = Model([words_max, train_y.shape[1]], words_size)
    model.set_model(128)
    model.compile()
    model.summary()
    # model.model.load_weights('./models/checkpoint.h5')  # 마지막 fine-tune 모델 가중치 로드
    model.fit(train_X, train_y, validation_data=(test_X, test_y), epochs=20, batch_size=64)
    model.save_model('full_dataset')
