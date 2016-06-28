# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 02:27:28 2016

@author: yash

TODO
1 : Word2Vec/GloVe word embedding lookup directly inside tensorflow
2 : Beam search instad of argmax

"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.models.rnn import rnn_cell

epoch = 100
lstm_size = 128
num_steps = 50
batch_size = 1
number_of_layers = 1
drop_prob = 0.5
vocab_size = 100 #TODO


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
# Placeholder for the inputs in a given iteration.
words = tf.placeholder(tf.int32, [batch_size, num_steps + 1]) #use this for both input and target
keep_prob = tf.placeholder(tf.float32)
lstm = rnn_cell.BasicLSTMCell(lstm_size)
stacked_lstm = rnn_cell.MultiRNNCell([lstm] * number_of_layers)

#Define readout variables
w_out = weight_variable([lstm_size, vocab_size])
b_out = bias_variable([vocab_size])
# Initial state of the LSTM memory.
initial_state = state = tf.zeros([batch_size, lstm.state_size])
loss = 0.0
for i in range(num_steps):
    # The value of state is updated after processing each batch of words.
    output, state = stacked_lstm(words[:, i], state)

    #add dropout to the output   
    out_drop  = tf.dropout(output, keep_prob)
    
    #calculate output probs
    logits = tf.matmul(out_drop, w_out) + b_out
    probabilities = tf.nn.softmax(logits)
    
    #evaluate loss
    loss += tf.reduce_mean(-tf.reduce_sum(words[:, i+1] * tf.log(probabilities), reduction_indices=[1]))

final_state = state  
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
 
sess = tf.Session()  
sess.run(tf.initialize_all_variables())
  
# A numpy array holding the state of LSTM after each batch of words.
updated_state = initial_state.eval()
total_loss = 0.0

for i in range(epoch):
    for current_batch_of_words in words_in_dataset:
        updated_state, _ = sess.run([final_state, train_step],
            # Initialize the LSTM state from the previous iteration.
            feed_dict={initial_state: updated_state, words: current_batch_of_words, keep_prob: drop_prob})
    #total_loss += current_loss
    #calculate accuracies accuracy.eval(feed_dict={initial_state: updated_state, words: current_batch_of_words, keep_prob: 1.0)
    
    
    
# embedding_matrix is a tensor of shape [vocabulary_size, embedding size]
word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)

sess.close()