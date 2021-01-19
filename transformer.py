import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds  
from tensorflow.keras.models import Sequential
from utils import *
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Embedding

class TokenEncoder:
    def __init__(self, tokenizer1, tokenizer2):
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2

    def encode(self, lang1, lang2):
        #adding tokens for start and end as well, vocab size is idx for start, vocab size + 1 is idx for end
        lang1 = [self.tokenizer1.vocab_size]+self.tokenizer1.encode(lang1.numpy())+[self.tokenizer1.vocab_size+1]
        lang2 = [self.tokenizer2.vocab_size]+self.tokenizer2.encode(lang2.numpy())+[self.tokenizer2.vocab_size+1]
        return lang1, lang2 

    def tf_encode(self, lang1, lang2):
        #we wrap the encode function as a tensorflow py_function so that it can be used together on each element of the dataset
        result_1, result_2 = tf.py_function(self.encode, [lang1, lang2], [tf.int64, tf.int64])
        result_1.set_shape([None])
        result_2.set_shape([None])
        #set shape to None allows any shape for the particular axis
        return result_1, result_2

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
        self.ffn = Utils.point_wise_feed_forward_nn(d_model, hidden)

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

        self.ffn = Utils.point_wise_feed_forward_nn(d_model, hidden)

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

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, hidden, input_vocab_size, target_vocab_size, t_input, t_target, dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, hidden, input_vocab_size, t_input, dropout_rate)
        self.decoder = Encoder(num_layers, d_model, num_heads, hidden, target_vocab_size, t_target, dropout_rate)

        self.final_layer = Dense(target_vocab_size)

    def forward(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder.forward(inp, training, enc_padding_mask)

        dec_output, attention_weights = self.decoder.forward(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)
        return final_output, attention_weights

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

if __name__=="__main__":
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)
    train, val = examples['train'], examples['validation']

    tokenizer1, tokenizer2 = initialize(train, val)

    BUFFER_SIZE = 20000
    BATCH_SIZE = 64
    MAX_LENGTH = 40

    tokenencode = TokenEncoder(tokenizer1, tokenizer2)

    train_dataset = train.map(tokenencode.tf_encode)
    train_dataset = train_dataset.filter(filter_max_length)

    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = val.map(tokenencode.tf_encode)
    val_dataset = val_dataset.filter(filter_max_length)

    num_layers = 4
    d_model = 128
    hidden = 512
    num_heads = 8
    #depth will be 16

    input_vocab_size = tokenizer1.vocab_size + 2
    target_vocab_size = tokenizer2.vocab_size + 2
    dropout_rate = 0.1