import numpy as np
import tensorflow as tf


def position_encoding(pos, i, embedding_size):
    if i % 2 == 0:
        return np.sin(pos / (10000 ** (2 * i / embedding_size)))
    return np.cos(pos / (10000 ** (2 * i / embedding_size)))


def tokenize(tokens, embeddings, padding_size):
    embeddings_size = len(embeddings["<start>"])
    positions_encoding = np.array(
        [[position_encoding(i, j, embeddings_size) for j in range(embeddings_size)] for i in range(padding_size)])
    out = []
    for token in tokens:
        out.append(embeddings[token])
    out = np.array(out)
    out = tf.add(out, positions_encoding[:out.shape[0]][:out.shape[1]])
    paddings = [[0, padding_size - out.shape[0]], [0, 0]]
    out = tf.pad(out, paddings)
    return np.array(out)


def to_onehot(token, embeddings):
    embeddings_list = list(embeddings.keys())
    dict_size = len(embeddings_list)
    idx = embeddings_list.index(token)
    return np.array([1 if i == idx else 0 for i in range(dict_size)])
