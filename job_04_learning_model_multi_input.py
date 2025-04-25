import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from job_04_customer_layer import TransformerEncoder

# 입력 설정
title_max_len, plot_max_len = 20, 327
title_word_size, plot_word_size = 10279, 33137

# 데이터 로드
X_train = np.load('./crawling_data/X_train_wordsize_{}_{}.npy'.format(title_word_size, plot_word_size), allow_pickle=True)
X_test = np.load('./crawling_data/X_test_wordsize_{}_{}.npy'.format(title_word_size, plot_word_size), allow_pickle=True)
y_train = np.load('./crawling_data/y_train_wordsize_{}_{}.npy'.format(title_word_size, plot_word_size), allow_pickle=True)
y_test = np.load('./crawling_data/y_test_wordsize_{}_{}.npy'.format(title_word_size, plot_word_size), allow_pickle=True)

num_classes = y_train.shape[1]

# 입력 분리
title_input = X_train[:, :title_max_len]
plot_input = X_train[:, title_max_len:title_max_len + plot_max_len]

title_test = X_test[:, :title_max_len]
plot_test = X_test[:, title_max_len:title_max_len + plot_max_len]

# === Title 인코더 ===
title_inputs = Input(shape=(title_max_len,))
title_embed = Embedding(title_word_size, 128)(title_inputs)
title_transformer = TransformerEncoder(64, 2, 128, embedding_dim=128)(title_embed)
title_encoded = GlobalAveragePooling1D()(title_transformer)

# === Plot 인코더 ===
plot_inputs = Input(shape=(plot_max_len,))
plot_embed = Embedding(plot_word_size, 128)(plot_inputs)
plot_transformer = TransformerEncoder(64, 2, 128, embedding_dim=128)(plot_embed)
plot_encoded = GlobalAveragePooling1D()(plot_transformer)

# === 병합 및 출력 ===
merged = Concatenate()([title_encoded, plot_encoded])
dense = Dense(128, activation='relu')(merged)
output = Dense(num_classes, activation='softmax')(dense)

model = Model(inputs=[title_inputs, plot_inputs], outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint('./models/transformer_multifeature.h5', monitor='val_accuracy', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

fit_hist = model.fit(
    [title_input, plot_input], y_train,
    epochs=20, batch_size=64,
    validation_data=([title_test, plot_test], y_test),
    callbacks=[checkpoint, early_stop]
)

score = model.evaluate([title_test, plot_test], y_test, verbose=0)
print("Test Accuracy:", score[1])

plt.plot(fit_hist.history['val_accuracy'], label='val accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()
