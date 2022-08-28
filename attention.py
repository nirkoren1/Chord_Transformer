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
        if self.mask:
            try:
                mask_ahead = np.array([[[0. if i >= j else -1e20 for j in range(QK.shape[2])] for i in range(QK.shape[1])] for k in range(QK.shape[0])])
            except IndexError:
                mask_ahead = np.array([[0. if i >= j else -1e20 for j in range(QK.shape[1])] for i in range(QK.shape[0])])
            mask_ahead = tf.cast(mask_ahead, tf.float32)
            QK = tf.add(QK, mask_ahead)
        mask = tf.cast(tf.math.equal(QK, 0), tf.float32)
        mask = tf.squeeze(mask[:, tf.newaxis, tf.newaxis, :])
        QK = tf.add(QK, tf.scalar_mul(-1e20, mask))
        QK = back.softmax(QK, axis=-1)
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
        self.normalize = LayerNormalization(axis=-1, center=True, scale=True, epsilon=0.0001)
        self.fc = Dense(embedding_size)
        self.dropout = Dropout(dropout_rate)

    def feed_forward(self, q, k, v, training=False):
        query = tf.split(q, self.h, axis=-1)
        keys = tf.split(k, self.h, axis=-1)
        value = tf.split(v, self.h, axis=-1)
        results = [head.feed_forward(self.query_weights[i](query[i]), self.keys_weights[i](keys[i]), self.value_weights[i](value[i])) for i, head in enumerate(self.heads)]
        result = tf.concat(results, -1)
        result = self.fc(result)
        if training:
            result = self.dropout(result)
        v = tf.cast(v, tf.float32)
        result = tf.add(result, v)
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
