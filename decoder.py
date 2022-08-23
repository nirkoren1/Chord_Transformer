from abc import ABC

import keras

from attention import MultiHeadAttention
from feedforward import FeedForward


class Decoder(keras.Model, ABC):
    def __init__(self, embedding_size, h, N=6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N
        self.multi_heads_masked = [MultiHeadAttention(embedding_size, h, mask=True) for i in range(N)]
        self.multi_heads = [MultiHeadAttention(embedding_size, h) for i in range(N)]
        self.feed_forwards = [FeedForward(embedding_size) for i in range(N)]

    def feed_forward(self, input_, q_encoder, k_encoder):
        result = input_
        for i in range(self.N):
            result = self.multi_heads_masked[i].feed_forward(result, result, result)
            result = self.multi_heads[i].feed_forward(q_encoder, k_encoder, result)
            result = self.feed_forwards[i].feed_forward(result)
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
    result = decoder.feed_forward(y, result, result)
    print(result)