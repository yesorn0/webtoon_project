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

title_model = Sequential()
title_model.add(Embedding(title_word_size, 256))
title_model.build(input_shape=(None, title_max_len))
title_model.add(Conv1D(32, kernel_size=8, padding='same', activation='relu'))
title_model.add(MaxPool1D(pool_size = 4))
title_model.add(LSTM(128, activation='tanh', return_sequences=True))
title_model.add(Dropout(0.3))
title_model.add(LSTM(64, activation='tanh', return_sequences=True))
title_model.add(Dropout(0.3))
title_model.add(LSTM(32, activation='tanh'))
title_model.add(Dense(128, activation='relu'))
title_model.add(Dense(y_train.shape[1], activation='softmax'))

title_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
title_model.summary()

# 저장 경로 설정
checkpoint_path = './models/title_lstm.h5'
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# 훈련
fit_hist = title_model.fit(title_input, y_train, epochs=10, batch_size=32,
                           validation_data=(title_test, y_test), callbacks=[checkpoint, early_stop])

score = title_model.evaluate(title_test, y_test, verbose=0)
print('Final test set accuracy', score[1])

plt.plot(fit_hist.history['val_accuracy'], label='val accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.savefig('./models/pretrained_model_{}_accuracy.jpg'.format(score[1]))
plt.show()
