from abc import ABC
import keras
import numpy as np
from encoder import Encoder
from decoder import Decoder
import tensorflow as tf
from utils import tokenize, padding_mask, look_ahead_mask


class Transformer(keras.Model, ABC):
    def __init__(self, embedding_size, h, dict_size, padding_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_size = padding_size
        self.encoder = Encoder(embedding_size, h)
        self.decoder = Decoder(embedding_size, h, dict_size)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        self.epoch_counter = 0
        self.accuracies = []
        self.losses_ = []

    def feed_forward(self, x, y, training=False):
        x_mask = padding_mask(x)
        encoder_output = self.encoder.feed_forward(x, x_mask, training)
        prediction = self.decoder.feed_forward(y, encoder_output, tf.math.maximum(padding_mask(y), look_ahead_mask(y)), x_mask, training)
        return prediction

    def loss_function(self, target, pred):
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss_ = self.loss_object(tf.squeeze(target), pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    def accuracy_function(self, real, pred):
        accuracies = tf.equal(real, tf.argmax(pred, axis=-1))

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    def learn(self, encoder_input, decoder_input, true_output, print_acc, reset_metrics_every=50):
        self.train_accuracy.reset_states()
        self.train_loss.reset_states()
        encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.float32)
        decoder_input = tf.convert_to_tensor(decoder_input, dtype=tf.float32)
        true_output = tf.convert_to_tensor(true_output, dtype=tf.int64)
        with tf.GradientTape(persistent=True) as tape:
            guess = self.feed_forward(encoder_input, decoder_input, training=True)
            # true_output = tf.reshape(true_output, (-1, true_output.shape[0]))
            loss = self.loss_function(true_output, guess)
            # loss = self.loss_object(true_output, guess)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.train_accuracy(self.accuracy_function(true_output, guess))
        self.train_loss(loss)
        self.accuracies.append(self.get_acc())
        self.losses_.append(self.get_loss())
        if print_acc:
            print("loss: ", self.get_loss_moving_avg(), "acc: ", self.get_acc_moving_avg())
        self.epoch_counter += 1

    def complete(self, input_, embeddings, max_output_len=40):
        dictionary_list = list(embeddings.keys())
        encoder_input = tokenize(input_, embeddings, self.padding_size)
        decoder_output_list = [input_[-1]]
        decoder_output_tokenize = tokenize(decoder_output_list, embeddings, self.padding_size)
        for _ in range(max_output_len):
            prediction = self.feed_forward(encoder_input, decoder_output_tokenize)[0]
            print(sorted(np.array(prediction))[-5:])
            print(np.max(prediction))
            print(np.argmax(prediction))
            prediction = dictionary_list[np.argmax(prediction)]
            print(prediction, end=" ")
            decoder_output_list += [prediction]
            decoder_output_tokenize = tokenize(decoder_output_list, embeddings, self.padding_size)
            # if prediction == "<end>":
            #     break

    def get_acc(self):
        return np.array(self.train_accuracy.result())

    def get_loss(self):
        return np.array(self.train_loss.result())

    def get_acc_moving_avg(self, window_size=50):
        if len(self.accuracies) < window_size:
            return 0
        return sum(self.accuracies[-window_size:]) / window_size

    def get_loss_moving_avg(self, window_size=50):
        if len(self.losses_) < window_size:
            return 0
        return sum(self.losses_[-window_size:]) / window_size

    def __call__(self, *args, **kwargs):
        self.feed_forward(args[0], args[1])
