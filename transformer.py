import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds  
from tf.keras.models import Sequential
from utils import *
from tf.keras.layers import Dense, LayerNormalization, Dropout, Embedding

class MultiHeadAttention:
    def __init__(self, d_model, heads):
        self.d_model = d_model
        self.num_heads = heads

        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads
        self.wq = Dense(self.d_model)
        self.wk = Dense(self.d_model)
        self.wv = Dense(self.d_model)

    def scaled_dot_product_attention(self, q, k, v, mask):
        #q, k must have same depth 
        #k, v mus have same seq_len
        #q dimension - (..., seq_len_q, dk)
        #k dimension - (..., seq_len_k, dk)
        #v dimension - (..., seq_len_k, dv)

        #calculating scaled dot product of keys and queries
        matmul_qk = tf.matmul(q, k, transpose_b = True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled = matmul_qk/tf.math.sqrt(dk)

        #in decoder we might not want the future values to get revealed due to attention
        #hence we mask the future values, by setting to -INF
        if mask is not None:
            scaled += (mask * -1e9)
        
        attention = tf.nn.softmax(scaled, axis=-1)
        output = tf.matmul(attention, v)

        return output, attention

    def split_heads(self, x, batch_size):
        #-1 will be the seq_len
        #reshape to make (..., seq_len, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0,2,1,3])

    def forward(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # making all q, k, v of shape (batch_size, seq_len, d_model)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        #making shape (batch_size, seq_len, num_heads, depth)
        scaled_attention = tf.transpose([0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = Dense(concat_attention)
        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, hidden, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_nn(d_model, hidden)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def forward(self, x, training, mask):
        output, attention_weights = self.mha.forward(x, x, x, mask)
        attn_output = self.dropout1(output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_head, hidden, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_head)
        self.mha2 = MultiHeadAttention(d_model, num_head)

        self.ffn = point_wise_feed_forward_nn(d_model, hidden)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)

    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights1 = self.mha1.forward(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, attn_weights2 = self.mha2.forward(enc_output, enc_output, x, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights1, attn_weights2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layer, d_model, num_heads, hidden, input_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layer

        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, hidden, dropout_rate) for _ in range(self.num_layers)]

        self.dropout = Dropout(dropout_rate)

    def forward(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, hidden, target_vocab_size, maximum_position_encoding, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, hidden, dropout_rate) for _ in range(self.num_layers)]
        self.dropout = Dropout(dropout_rate)

    def forward(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training = training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        return x, attention_weights

if __name__=="__main__":
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    train, val = examples['train'], examples['val']

    '''
    #instantiating a transformer and loading dataset
    transformer = Transformer(train, val)
    train_ds = train.map(transformer.tf_encode)
    train_ds = train_ds.filter(tranformer.filter_max_length)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(transformer.buffer_size).padded_batch(transformer.batch_size)

    val_ds = val.map(transformer.tf_encode)
    val_ds = val_ds.filter(transformer.filter_max_length).padded_batch(transformer.batch_size)
    '''

