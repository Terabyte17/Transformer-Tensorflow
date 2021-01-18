import numpy as np
import matplotlib.pyplot as plt
import time

import tensorflow as tf 
import tensorflow_datasets as tfds
from tf.keras.layers import Dense
from tf.keras.models import Sequential

class Utils:
    def __init__(self, train, val, max_length=40, buffer_size=20000, batch_size=64):
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        #making the tokenizer from corpus
        #the subword text encoder encodes unknown words by breaking them further, and hence is able to handle OOV words

        self.tokenizer1 = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((lang1.numpy for lang1, lang2 in train), target_vocab_size = 2**13)
        self.tokenizer2 = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((lang2.numpy for lang1, lang2 in train), target_vocab_size = 2**13)
        
    def encode(self, lang1, lang2):
        #adding tokens for start and end as well, vocab size is idx for start, vocab size + 1 is idx for end
        lang1 = [self.tokenizer1.vocab_size]+self.tokenizer1.encode(lang1.numpy())+[self.tokenizer1.vocab_size+1]
        lang2 = [self.tokenizer2.vocab_size]+self.tokenizer2.encode(lang2.numpy())+[self.tokenizer2.vocab_size+1]

    def tf_encode(self, lang1, lang2):
        #we wrap the encode function as a tensorflow py_function so that it can be used together on each element of the dataset
        result_1, result_2 = tf.py_function(encode, [lang1, lang2], [tf.int64, tf.int64])
        result_1.set_shape([None])
        result_2.set_shape([None])
        #set shape to None allows any shape for the particular axis

    def filter_max_length(self, x, y):
        return tf.logical_and(tf.size(x)<=self.max_length, tf.size(y)<=self.max_length)

    def get_angles(self, pos, i):
        angle_rates = 1/np.power(10000, (2*(i//2))/np.float32(self.d_model))
        return pos*angle_rates

    def positional_encoding(self, position):
        #here the i is passed till self.d_model - dimension of the input
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(self.d_model)[np.newaxis,:])

        #sin encoding to even, cos to odd
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, :, :]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def visualize_pos_encoding(self, position_encoding):
        plt.pcolormesh(position_encoding[0], cmap='RdBu')
        plt.xlabel('Depth')
        plt.xlim((0,512))
        plt.ylabel('Position')
        plt.colorbar()
        plt.show()
        time.sleep(5)
        plt.close()

    def create_padding_mask(self, seq):
        #checks which all elements in the seq are 0
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        #band part will give lower triangle and subtracting from one will give mask
        return mask

    def point_wise_feed_forward_nn(self, d_model, hidden):
        model = Sequential()
        model.add(Dense(hidden, activation="relu"))
        model.add(Dense(d_model))
        return model