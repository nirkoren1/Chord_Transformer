from abc import ABC
import keras
import numpy as np
from encoder import Encoder
from decoder import Decoder
import tensorflow as tf
from tokenizer import tokenize


class Transformer(keras.Model, ABC):
    def __init__(self, embedding_size, h, dict_size, padding_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_size = padding_size
        self.encoder = Encoder(embedding_size, h)
        self.decoder = Decoder(embedding_size, h, dict_size)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    def feed_forward(self, x, y, training=False):
        encoder_output = self.encoder.feed_forward(x, training)
        prediction = self.decoder.feed_forward(y, encoder_output, training)
        return prediction

    def loss_function(self, target, pred):
        mask = tf.math.logical_not(tf.math.equal(target, 1 / 256))
        loss_ = self.loss_object(tf.squeeze(target), pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def learn(self, encoder_input, decoder_input, true_output, print_acc):
        encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.float32)
        decoder_input = tf.convert_to_tensor(decoder_input, dtype=tf.float32)
        true_output = tf.convert_to_tensor(true_output, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            guess = self.feed_forward(encoder_input, decoder_input, training=True)
            # true_output = tf.reshape(true_output, (-1, true_output.shape[0]))
            loss = self.loss_function(true_output, guess)
            # loss = tf.losses.SparseCategoricalCrossentropy()(true_output, guess)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_accuracy(true_output, guess)
        if print_acc:
            print("loss: ", loss, self.train_accuracy.result())

    def complete(self, input_, embeddings):
        dictionary_list = list(embeddings.keys())
        encoder_input = tokenize(input_, embeddings, self.padding_size)
        decoder_output_list = [input_[-1]]
        decoder_output_tokenize = tokenize(decoder_output_list, embeddings, self.padding_size)
        prediction = input_[-1]
        while prediction != "<end>":
            prediction = self.feed_forward(encoder_input, decoder_output_tokenize)[0]
            print(sorted(np.array(prediction))[-5:])
            print(np.max(prediction))
            print(np.argmax(prediction))
            prediction = dictionary_list[np.argmax(prediction)]
            print(prediction, end=" ")
            decoder_output_list += [prediction]
            decoder_output_tokenize = tokenize(decoder_output_list, embeddings, self.padding_size)

    def get_acc(self):
        return np.array(self.train_accuracy.result())


if __name__ == '__main__':
    x = np.array([[np.random.random() for i in range(16)] for j in range(7)])
    y = np.array([[np.random.random() for i in range(16)] for j in range(5)])
    transformer = Transformer(16, 4, 100, 20)
    print(transformer.feed_forward(x, y))
