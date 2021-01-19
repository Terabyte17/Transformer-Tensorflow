import numpy as np
import matplotlib.pyplot as plt
import time

import tensorflow as tf 
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def initialize(train, val, max_length=40, buffer_size=20000, batch_size=64):
    max_length = max_length
    buffer_size = buffer_size
    batch_size = batch_size
    #making the tokenizer from corpus
    #the subword text encoder encodes unknown words by breaking them further, and hence is able to handle OOV words

    tokenizer1 = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((lang1.numpy() for lang1, lang2 in train), target_vocab_size = 2**13)
    tokenizer2 = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((lang2.numpy() for lang1, lang2 in train), target_vocab_size = 2**13)

    return tokenizer1, tokenizer2
    
def filter_max_length(x, y, max_length=40):
    return tf.logical_and(tf.size(x)<=max_length, tf.size(y)<=max_length)

def get_angles(pos, i, d_model):
    angle_rates = 1/np.power(10000, (2*(i//2))/np.float32(d_model))
    return pos*angle_rates

def positional_encoding(position, d_model):
    #here the i is passed till d_model - dimension of the input
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis,:], d_model)

    #sin encoding to even, cos to odd
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, :, :]
    return tf.cast(pos_encoding, dtype=tf.float32)

def visualize_pos_encoding(position_encoding):
    plt.pcolormesh(position_encoding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0,512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()
    time.sleep(5)
    plt.close()

def create_padding_mask(seq):
    #checks which all elements in the seq are 0
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    #band part will give lower triangle and subtracting from one will give mask
    return mask

def point_wise_feed_forward_nn(d_model, hidden):
    model = Sequential()
    model.add(Dense(hidden, activation="relu"))
    model.add(Dense(d_model))
    return model