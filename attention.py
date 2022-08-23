from abc import ABC
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras import backend as back


class AttentionHead(keras.Model, ABC):
    def __init__(self, embedding_size, h):
        super(AttentionHead, self).__init__()
        dk = int(embedding_size / h)
        self.dk = dk
        self.value = Dense(dk)
        self.query = Dense(dk)
        self.keys = Dense(dk)

    def feed_forward(self, input_):
        value = self.value(input_)
        query = self.query(input_)
        keys = self.keys(input_)
        QK = tf.linalg.matmul(query, keys, transpose_b=True)
        QK = tf.scalar_mul(1 / self.dk, QK)
        QK = back.softmax(QK, axis=-1)
        QKV = tf.linalg.matmul(QK, value)
        return QKV


class MultiHeadAttention(keras.Model, ABC):
    def __init__(self, embedding_size, h: int):
        super(MultiHeadAttention, self).__init__()
        self.heads = [AttentionHead(embedding_size, h) for i in range(h)]

    def feed_forward(self, input_):
        results = [head.feed_forward(input_) for head in self.heads]
        result = tf.concat(results, 1)
        return result
