import pickle
import random

from transformer import Transformer
import tensorflow as tf

with open("word2vec/embeddings.pickle", 'rb') as f:
    embeddings = pickle.load(f)
with open("data_clean/data.pickle", 'rb') as f:
    data = pickle.load(f)

embeddings_size = 256
padding_size = 300
dict_size = embeddings.size
h = 8
transformer = Transformer(embeddings_size, h, dict_size, padding_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=0, beta_1=0.9, beta_2=0.98, epsilon=1e-09)
transformer.compile(loss='crossentropy', optimizer=optimizer, metrics=['accuracy'])
batch_size = 16
x_size = 100_000

x = []
y = []
for i in range(x_size):
    rand_row = random.choice(data["training_data"].to_list())
    start_idx = rand_row.index("<start>")
print(data.head())