from abc import ABC
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from keras import backend as back
import numpy as np


class AttentionHead(keras.Model, ABC):
    def __init__(self, embedding_size, h, mask):
        super(AttentionHead, self).__init__()
        dk = int(embedding_size / h)
        self.dk = dk
        self.mask = mask

    def feed_forward(self, q, k, v):
        QK = tf.linalg.matmul(q, k, transpose_b=True)
        QK = tf.scalar_mul(1 / self.dk, QK)
        mask = tf.cast(tf.math.equal(QK, 0), tf.float32)
        mask = mask[:, tf.newaxis, tf.newaxis, :]
        QK = tf.add(QK, tf.scalar_mul(-np.inf, mask))
        if self.mask == "look_ahead":
            mask_ahead = np.array([[[0. if i >= j else -np.inf for j in range(QK.shape[2])] for i in range(QK.shape[1])] for k in range(QK.shape[0])])
            mask = tf.add(mask, mask_ahead)
        mask = tf.cast(mask, tf.float32)
        QK = tf.add(QK, mask)
        QK = back.softmax(QK, axis=-1)
        QK = tf.where(tf.math.is_nan(QK), tf.zeros_like(QK), QK)
        QKV = tf.linalg.matmul(QK, v)
        return QKV


class MultiHeadAttention(keras.Model, ABC):
    def __init__(self, embedding_size, h: int, mask=False, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        dk = int(embedding_size / h)
        self.dk = dk
        self.value_weights = [Dense(dk) for _ in range(h)]
        self.query_weights = [Dense(dk) for _ in range(h)]
        self.keys_weights = [Dense(dk) for _ in range(h)]
        self.heads = [AttentionHead(embedding_size, h, mask) for i in range(h)]
        self.normalize = LayerNormalization(axis=1, center=True, scale=True, epsilon=0.0001)
        self.fc = Dense(embedding_size)
        self.dropout = Dropout(dropout_rate)

    def feed_forward(self, q, k, v, training=False):
        query = tf.split(q, self.h, axis=2)
        keys = tf.split(k, self.h, axis=2)
        value = tf.split(v, self.h, axis=2)
        results = [head.feed_forward(self.query_weights[i](query[i]), self.keys_weights[i](keys[i]), self.value_weights[i](value[i])) for i, head in enumerate(self.heads)]
        result = tf.concat(results, 1)
        result = self.fc(result)
        if training:
            result = self.dropout(result)
        v = tf.cast(v, tf.float32)
        result = tf.add(result, v)
        result = self.normalize(result)
        return result


if __name__ == '__main__':
    x = np.array([[0.98, 1.28, 0.41, 0.27],
                  [0.52, 0.01, 2.06, 0.27],
                  [2.22, 0.27, 0.1, 0.41],
                  [0.99, 1, 0.11, 0.27],
                  [0.52, 0.01, 0.33, 2.06],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    y = np.array([[1.2, 2.3, 1.4, 0.2],
                  [0.2, 0.5, 0.6, 0.8],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    at = AttentionHead(4, 2, True)
    print(at.feed_forward(y, x, x))
