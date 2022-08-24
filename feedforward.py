from abc import ABC
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from keras import backend as back


class FeedForward(keras.Model, ABC):
    def __init__(self, embedding_size, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = Dense(embedding_size * 4, activation='relu')
        self.fc2 = Dense(embedding_size)
        self.normalize = LayerNormalization(axis=1, center=True, scale=True, epsilon=0.0001)
        self.dropout = Dropout(dropout_rate)

    def feed_forward(self, input_, training=False):
        result = self.fc1(input_)
        result = self.fc2(result)
        if training:
            self.dropout(result)
        result = tf.add(result, input_)
        result = self.normalize(result)
        return result


if __name__ == '__main__':
    import numpy as np

    x = np.array([[0.98, 1.28, 0.41, 0.27, 0.41],
                  [0.52, 0.01, 2.06, 0.27, 0.33],
                  [2.22, 0.27, 0.1, 0.41, 2.06],
                  [0.99, 1, 0.11, 0.27, 0.33],
                  [0.52, 0.01, 0.33, 2.06, 0.52],
                  [0.10, 2.06, 0.73, 0.27, 0.41],
                  [0.33, 0.01, 0.13, 0.27, 1.28]])
    f = FeedForward(5)
    print(f.feed_forward(x))