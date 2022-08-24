from abc import ABC
import keras
import numpy as np
from encoder import Encoder
from decoder import Decoder
import tensorflow as tf


def position_encoding(pos, i, embedding_size):
    if i % 2 == 0:
        return np.sin(pos / (10000 ** (2 * i / embedding_size)))
    return np.cos(pos / (10000 ** (2 * i / embedding_size)))


class Transformer(keras.Model, ABC):
    def __init__(self, embedding_size, h, dict_size, padding_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_size = padding_size
        self.encoder = Encoder(embedding_size, h)
        self.decoder = Decoder(embedding_size, h, dict_size)
        self.position_encoding = np.array([[position_encoding(i, j, embedding_size) for j in range(embedding_size)] for i in range(padding_size)])

    def feed_forward(self, x, y, training=False):
        x = tf.add(x, self.position_encoding[:x.shape[0]][:x.shape[1]])
        y = tf.add(y, self.position_encoding[:y.shape[0]][:y.shape[1]])
        paddings = [[0, self.padding_size - x.shape[0]], [0, 0]]
        x = tf.pad(x, paddings)
        paddings = [[0, self.padding_size - y.shape[0]], [0, 0]]
        y = tf.pad(y, paddings)
        encoder_output = self.encoder.feed_forward(x, training)
        prediction = self.decoder.feed_forward(y, encoder_output, training)
        return prediction


if __name__ == '__main__':
    x = np.array([[np.random.random() for i in range(16)] for j in range(7)])
    y = np.array([[np.random.random() for i in range(16)] for j in range(5)])
    transformer = Transformer(16, 4, 100, 20)
    print(transformer.feed_forward(x, y))
