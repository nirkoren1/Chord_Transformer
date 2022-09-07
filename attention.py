from abc import ABC
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from keras import backend as back
import numpy as np


class AttentionHead(keras.layers.Layer, ABC):
    def __init__(self, embedding_size, h):
        super(AttentionHead, self).__init__()
        dk = int(embedding_size / h)
        self.dk = dk

    def feed_forward(self, q, k, v, mask):
        QK = tf.linalg.matmul(q, k, transpose_b=True)
        QK = tf.scalar_mul(1 / np.sqrt(self.dk), QK)
        QK = tf.add(QK, tf.scalar_mul(-1e9, mask))
        QK = tf.nn.softmax(QK, axis=-1)
        QKV = tf.linalg.matmul(QK, v)
        return QKV


class MultiHeadAttention(keras.layers.Layer, ABC):
    def __init__(self, embedding_size, h: int, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        dk = int(embedding_size / h)
        self.dk = dk
        self.value_weights = [Dense(dk) for _ in range(h)]
        self.query_weights = [Dense(dk) for _ in range(h)]
        self.keys_weights = [Dense(dk) for _ in range(h)]
        self.heads = [AttentionHead(embedding_size, h) for i in range(h)]
        self.normalize = LayerNormalization(axis=-1, center=True, scale=True, epsilon=0.0001)
        self.fc = Dense(embedding_size)
        self.dropout = Dropout(dropout_rate)

    def feed_forward(self, q, k, v, mask, training=False):
        query = tf.split(q, self.h, axis=-1)
        keys = tf.split(k, self.h, axis=-1)
        value = tf.split(v, self.h, axis=-1)
        results = [head.feed_forward(self.query_weights[i](query[i]), self.keys_weights[i](keys[i]), self.value_weights[i](value[i]), mask) for i, head in enumerate(self.heads)]
        result = tf.concat(results, -1)
        result = self.fc(result)
        if training:
            result = self.dropout(result)
        v = tf.cast(v, tf.float32)
        result = tf.add(result, v)
        if len(result.shape) == 2:
            result = tf.reshape(result, (1, result.shape[0], result.shape[1]))
        result = self.normalize(result)
        return result


if __name__ == '__main__':
    x = np.array([[np.random.random() if i < 50 else 0.0 for j in range(5)] for i in range(5)])
    print(x)
    y = np.array([[np.random.random() if i < 60 else 0.0 for j in range(256)] for i in range(256)])
    normalize1 = LayerNormalization(axis=-1, center=True, scale=True, epsilon=0.0001)
    normalize2 = LayerNormalization(axis=1, center=True, scale=True, epsilon=0.0001)
    print(normalize1(x))
    print(normalize2(x))
    # at = MultiHeadAttention(256, 8)
    # print(at.feed_forward(y, x, x))
