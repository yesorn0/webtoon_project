from keras.layers import LayerNormalization, Dense, Dropout
from keras.layers import MultiHeadAttention
from tensorflow.keras.layers import Layer
import tensorflow as tf


class TransformerEncoder(Layer):
    def __init__(self, head_size, num_heads, ff_dim, embedding_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)
        self.dropout1 = Dropout(dropout)
        self.norm1 = LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embedding_dim),  # <- 핵심!
        ])
        self.dropout2 = Dropout(dropout)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)
