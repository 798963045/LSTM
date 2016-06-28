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
from tensorflow.python.ops import rnn_cell
from data import Data

epoch = 100
lstm_size = 1024
num_steps = 25
batch_size = 16
number_of_layers = 2
max_grad_norm = 5
drop_prob = 0.5
vocab_size = 50 #TODO
encoding = 'Glove' #'One-hot'       

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Placeholder for the inputs in a given iteration.
#make words type int32 when using one hot vector
words = tf.placeholder(tf.float32, [batch_size, num_steps + 1, vocab_size]) #use this for both input and target
keep_prob = tf.placeholder(tf.float32)
lstm = rnn_cell.BasicLSTMCell(lstm_size)
stacked_lstm = rnn_cell.MultiRNNCell([lstm] * number_of_layers)
# Initial state of the LSTM memory.
#initial_state = state = tf.zeros([batch_size, lstm.state_size])
initial_state = state = stacked_lstm.zero_state(batch_size, tf.float32)
#Define readout variables
w_out = weight_variable([lstm_size, vocab_size])
b_out = bias_variable([vocab_size])


loss = 0.0
with tf.variable_scope("RNN"):
    for i in range(num_steps):
        if i>0: tf.get_variable_scope().reuse_variables()
        # The value of state is updated after processing each batch of words.
        output, state = stacked_lstm(words[:, i, :], state)
        #add dropout to the output   
        out_drop  = tf.nn.dropout(output, keep_prob)
        
        #outputs.append(out_drop)
        #calculate output probs
        logits = tf.matmul(out_drop, w_out) + b_out
          
        if encoding == 'Glove':
            #use tanh activation since target's range is [-1,1]
            #use euclidean loss since targets are NOT one hot vector 
            prob = tf.tanh(logits)
            loss += tf.reduce_mean(-tf.reduce_sum(tf.square(words[:,i+1, :] - prob), reduction_indices=[1]))
        
        else:
            probabilities = tf.nn.softmax(logits)    
            loss += tf.reduce_mean(-tf.reduce_sum(words[:, i+1, :] * tf.log(probabilities), reduction_indices=[1]))
 
final_state = state  

#_lr = tf.Variable(0.0, trainable=False)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),  max_grad_norm)
optimizer = tf.train.AdamOptimizer(1e-4)
train_step = optimizer.apply_gradients(zip(grads, tvars))

#train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()  
sess.run(tf.initialize_all_variables())
  
# A numpy array holding the state of LSTM after each batch of words.
updated_state = initial_state.eval(session = sess)
#total_loss = 0.0

folder = '/home/yash/Project/dataset/SoP_data/'
files = ['cs.txt', 'maths.txt', 'engg.txt', 'physics.txt', 'bio.txt', 'chem.txt', 'management.txt',  'design.txt',  'finance.txt', 'law.txt', 'literature.txt',  'others.txt']
data = Data(folder, files, batch_size, num_steps)
ctr = 0
while True:#for i in range(epoch):
    while data.has_more_data:
        d = data.get_next_batch()
        updated_state, l, _ = sess.run([final_state, loss, train_step],
            # Initialize the LSTM state from the previous iteration.
            feed_dict={initial_state: updated_state, words: d, keep_prob: drop_prob})
    
        if ctr%100 == 0:
            print("Error at batch %d: %0.5f" %(ctr, l) )
        
        ctr += 1
        #total_loss += current_loss
        #calculate accuracies accuracy.eval(feed_dict={initial_state: updated_state, words: current_batch_of_words, keep_prob: 1.0)
    data.has_more_data = True
    
    
# embedding_matrix is a tensor of shape [vocabulary_size, embedding size]
#word_embeddings = tf.nn.embedding_lookup(embedding_matrix, word_ids)

sess.close()