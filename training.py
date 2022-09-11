import pickle5
import random
from abc import ABC
import animate
import numpy as np
from transformer import Transformer
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

with open("word2vec/embeddings.pickle", 'rb') as f:
    embeddings = pickle5.load(f)
with open("data_clean/data.pickle", 'rb') as f:
    data = pickle5.load(f)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule, ABC):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def position_encoding(pos, i, embedding_size):
    if i % 2 == 0:
        return np.sin(pos / (10000 ** (i / embedding_size)))
    return np.cos(pos / (10000 ** (i / embedding_size)))


def tokenize(tokens):
    out = []
    for token in tokens:
        out.append(embeddings[token])
    out = np.array(out)
    out = tf.add(out, positions_encoding[:out.shape[0]][:out.shape[1]])
    paddings = [[0, padding_size - out.shape[0]], [0, 0]]
    out = tf.pad(out, paddings)
    return np.array(out)


def to_onehot(token):
    idx = embeddings_list.index(token)
    return np.array([1 if i == idx else 0 for i in range(dict_size)])


embeddings_size = 256
padding_size = 256
dict_size = len(embeddings)
print("dict size: ", dict_size)
h = 8
batch_size = 8
x_size = 20_000
validation_size = 1000
embeddings_list = list(embeddings.keys())
positions_encoding = np.array(
        [[position_encoding(i, j, embeddings_size) for j in range(embeddings_size)] for i in range(padding_size)])


encoder_input = []
decoder_input = []
true_y = []
print()
for i in range(x_size + validation_size):
    print("\r", i, end="")
    rand_row = random.choice(data["training_data"].to_list())
    start_idx = rand_row.index("<start>")
    cut1_idx = random.randint(start_idx + 1, min(len(rand_row) - 1, padding_size - 5))
    cut2_idx = random.randint(cut1_idx, min(len(rand_row) - 1, padding_size - 5))
    encoder_input.append(tokenize(rand_row[:cut1_idx]))
    decoder_input.append(tokenize(rand_row[cut1_idx - 1: cut2_idx]))
    true_y.append(np.argmax(to_onehot(rand_row[cut2_idx])))
print()


if __name__ == '__main__':
    epochs = 100_000
    transformer = Transformer(embeddings_size, 8, dict_size, padding_size)
    learning_rate = CustomSchedule(embeddings_size, 4000)
    transformer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9))
    best_acc = 0
    for epoch in range(epochs):
        print(epoch, end=" ")
        encoder_input_batch = random.sample(encoder_input, batch_size)
        decoder_input_batch = random.sample(decoder_input, batch_size)
        true_y_batch = random.sample(true_y, batch_size)
        transformer.learn(encoder_input_batch, decoder_input_batch, true_y_batch, True)
        animate.update(transformer.get_loss_moving_avg(), "loss_scores", 0)
        animate.update(transformer.get_acc_moving_avg(), "acc_scores", 1)
        if transformer.get_acc_moving_avg() > best_acc and epoch % 10 == 0:
            best_acc = transformer.get_acc_moving_avg()
            print(f"saving model with acc of {best_acc}")
            transformer.save_weights("weights")
