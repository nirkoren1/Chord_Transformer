from abc import ABC
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LayerNormalization
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
        if self.mask:
            mask = np.array([[0. if i >= j else -np.inf for j in range(QK.shape[1])] for i in range(QK.shape[0])])
            QK = tf.add(QK, mask)
        QK = back.softmax(QK, axis=-1)
        QKV = tf.linalg.matmul(QK, v)
        return QKV


class MultiHeadAttention(keras.Model, ABC):
    def __init__(self, embedding_size, h: int, mask=False):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        dk = int(embedding_size / h)
        self.dk = dk
        self.value = Dense(dk)
        self.query = Dense(dk)
        self.keys = Dense(dk)
        self.heads = [AttentionHead(embedding_size, h, mask) for i in range(h)]
        self.normalize = LayerNormalization(axis=1, center=True, scale=True, epsilon=0.0001)
        self.fc = Dense(embedding_size)

    def feed_forward(self, q, k, v):
        query = tf.split(self.query(q), self.h, axis=1)
        keys = tf.split(self.keys(k), self.h, axis=1)
        value = tf.split(self.value(v), self.h, axis=1)
        results = [head.feed_forward(query[i], keys[i], value[i]) for i, head in enumerate(self.heads)]
        result = tf.concat(results, 1)
        result = self.fc(result)
        result = tf.add(result, v)
        result = self.normalize(result)
        return result


if __name__ == '__main__':
    x = np.array([[0.98, 1.28, 0.41, 0.27],
                  [0.52, 0.01, 2.06, 0.27],
                  [2.22, 0.27, 0.1, 0.41],
                  [0.99, 1, 0.11, 0.27],
                  [0.52, 0.01, 0.33, 2.06],
                  [0.10, 2.06, 0.73, 0.27],
                  [0.33, 0.01, 0.13, 0.27]])
    x = tf.convert_to_tensor(x)
    x = tf.split(x, 2, axis=1)
    print(x[0])
    print(x[1])
    # at = AttentionHead(5, 1, True)
    # print(at.feed_forward(x))
