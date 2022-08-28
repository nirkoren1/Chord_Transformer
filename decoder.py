from abc import ABC
import keras
from keras.layers import Flatten, Dense
from attention import MultiHeadAttention
from feedforward import FeedForward
import tensorflow as tf
import numpy as np


class Decoder(keras.Model, ABC):
    def __init__(self, embedding_size, h, dict_size, N=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N
        self.multi_heads_masked = [MultiHeadAttention(embedding_size, h, mask=True) for i in range(N)]
        self.multi_heads = [MultiHeadAttention(embedding_size, h) for i in range(N)]
        self.feed_forwards = [FeedForward(embedding_size) for i in range(N)]
        self.flat = keras.layers.Flatten()
        self.fc = Dense(dict_size, activation='softmax')

    def feed_forward(self, input_, encoder_output, training):
        result = input_
        for i in range(self.N):
            result = self.multi_heads_masked[i].feed_forward(result, result, result, training)
            result = self.multi_heads[i].feed_forward(result, encoder_output, encoder_output, training)
            result = self.feed_forwards[i].feed_forward(result, training)
        try:
            result = tf.reshape(result, [result.shape[0], result.shape[1] * result.shape[2]])
        except IndexError:
            result = tf.reshape(result, [1, result.shape[0] * result.shape[1]])
        result = self.fc(result)
        return result


if __name__ == '__main__':
    import numpy as np
    from encoder import Encoder
    x = np.array([[np.random.random() for i in range(4)] for j in range(7)])
    y = np.array([[np.random.random() for i in range(4)] for j in range(10)])
    encoder = Encoder(4, 2)
    decoder = Decoder(4, 2)
    result = encoder.feed_forward(x)
    print(result)
    result = decoder.feed_forward(y, result)
    print(result)