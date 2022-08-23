from abc import ABC
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import random
import numpy as np
import pickle
from scipy.spatial import distance


class SkipGram(keras.Model, ABC):
    def __init__(self, input_dim, embedding_dim):
        super(SkipGram, self).__init__()
        self.fc1 = Dense(embedding_dim, input_dim=input_dim)
        self.fc2 = Dense(input_dim, activation='softmax')

    def feed_forward(self, input_):
        out = self.fc1(input_)
        out = self.fc2(out)
        return out

    def learn(self, x, y_true):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            y_guess = self.feed_forward(x)
            auto_loss = tf.losses.MSE(y_true, y_guess)
        gradient = tape.gradient(auto_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))

    def save_model(self, path):
        self.save_weights(path)
        print("Model Saved")


hash_chords = {}
df_one_hot = pd.read_pickle(r"C:\Users\Nirkoren\PycharmProjects\Chord_Transformer\data_clean\chords1Hot.pickle")
for idx, row in df_one_hot.iterrows():
    hash_chords[row["chords"]] = row["one_hot"]
df = pd.read_pickle(r"C:\Users\Nirkoren\PycharmProjects\Chord_Transformer\data_clean\data.pickle")
window_size = 7
x_size = 300_000
dic_size = len(hash_chords)


def create_data():
    x = []
    y = []
    while len(x) < x_size:
        print('\r', f"{len(x)}/{x_size}", end="")
        raw = random.choice(df["training_data"].tolist())
        center = random.randint(0, len(raw) - 1)
        for j in range(window_size):
            if j != int(window_size / 2):
                try:
                    y.append(np.array(hash_chords[raw[center - int((window_size - 1) / 2) + j]]))
                    x.append(np.array(hash_chords[raw[center]]))
                except IndexError as e:
                    pass
    x = np.array(x)
    y = np.array(y)
    with open("x.pickle", 'wb') as f:
        pickle.dump(x, f)
    with open("y.pickle", 'wb') as f:
        pickle.dump(y, f)


if __name__ == '__main__':
    # create_data()

    # train the network
    with open("x.pickle", 'rb') as f:
        x = pickle.load(f)
    with open("y.pickle", 'rb') as f:
        y = pickle.load(f)
    embed_size = 256
    batch_size = 16
    epochs = 1000
    model = SkipGram(dic_size, embed_size)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    for i in range(epochs):
        print('\r', f"epoch {i + 1}/{epochs} ", end="")
        batch = np.random.choice(x_size, batch_size)
        x_batch = x[batch]
        y_batch = y[batch]
        model.learn(x_batch, y_batch)
    print()

    embeddings_dict = {}
    for idx, word in enumerate(list(hash_chords.keys())):
        print("\r", f"{idx}/{dic_size}", end="")
        embeddings_dict[word] = np.array(model.fc1.weights[0])[idx]
    print()
    with open("embeddings.pickle", 'wb') as f:
        pickle.dump(embeddings_dict, f)

    # evaluate result
    with open("embeddings.pickle", 'rb') as f:
        embeddings = pickle.load(f)

    word1 = "C"
    similar_words = ["Dm", "Em", "F", "G", "Am"]
    not_similar_words = ["D", "E", "F#", "G#m", "A"]
    for word in similar_words:
        sim1 = 1 - distance.cosine(embeddings[word1], embeddings[word])
        print(word, sim1, end=" ")
    print()
    for word in not_similar_words:
        sim1 = 1 - distance.cosine(embeddings[word1], embeddings[word])
        print(word, sim1, end=" ")
