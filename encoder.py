from abc import ABC

import keras

from attention import MultiHeadAttention
from feedforward import FeedForward


class Encoder(keras.Model, ABC):
    def __init__(self, embedding_size, h, N=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N
        self.multi_heads = [MultiHeadAttention(embedding_size, h) for i in range(N)]
        self.feed_forwards = [FeedForward(embedding_size) for i in range(N)]

    def feed_forward(self, input_):
        result = input_
        for i in range(self.N):
            result = self.multi_heads[i].feed_forward(result, result, result)
            result = self.feed_forwards[i].feed_forward(result)
        return result


if __name__ == '__main__':
    import numpy as np

    x = np.array([[0.98, 1.28, 0.41, 0.27],
                  [0.52, 0.01, 2.06, 0.27],
                  [2.22, 0.27, 0.1, 0.41],
                  [0.99, 1, 0.11, 0.27],
                  [0.52, 0.01, 0.33, 2.06],
                  [0.10, 2.06, 0.73, 0.27],
                  [0.33, 0.01, 0.13, 0.27]])
    at = Encoder(4, 2)
    print(at.feed_forward(x))