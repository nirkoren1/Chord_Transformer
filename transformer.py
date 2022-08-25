from abc import ABC
import keras
import numpy as np
from encoder import Encoder
from decoder import Decoder
import tensorflow as tf


class Transformer(keras.Model, ABC):
    def __init__(self, embedding_size, h, dict_size, padding_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_size = padding_size
        self.encoder = Encoder(embedding_size, h)
        self.decoder = Decoder(embedding_size, h, dict_size)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    def feed_forward(self, x, y, training=False):
        encoder_output = self.encoder.feed_forward(x, training)
        prediction = self.decoder.feed_forward(y, encoder_output, training)
        return prediction

    def loss_function(self, target, pred):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        print(tf.squeeze(target))
        print(pred)
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
            # loss = self.loss_function(true_output, guess)
            loss = tf.losses.SparseCategoricalCrossentropy()(true_output, guess)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_accuracy(true_output, guess)
        if print_acc:
            print("loss: ", loss)


if __name__ == '__main__':
    x = np.array([[np.random.random() for i in range(16)] for j in range(7)])
    y = np.array([[np.random.random() for i in range(16)] for j in range(5)])
    transformer = Transformer(16, 4, 100, 20)
    print(transformer.feed_forward(x, y))
