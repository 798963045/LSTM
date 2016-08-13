# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 14:10:48 2016

Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import re
from collections import Counter

def preprocess(d):
    d = re.sub(r' +',r' ',d.upper()) #remove extra spaces
    #remove non-recognisable char
    #return re.sub(r'[^\x00-\x7f]',r'',d.upper()) 
    # print ([ord(c) for c in ['!', ':', '.', ',', '\"', "\'", ' ']])
    stripped = [c for c in d if (64<ord(c)<91 or 96<ord(c)<123 or 47<ord(c)<58 or ord(c) in [33, 58, 46, 44, 34, 39, 32]) ]
    return ''.join(stripped)
    
# data I/O
files = ['wiki_text.txt']#['cs.txt', 'maths.txt', 'engg.txt', 'physics.txt']#, 'bio.txt', 'chem.txt', 'management.txt',  'design.txt',  'finance.txt', 'law.txt', 'literature.txt',  'others.txt']
path = '/home/yash/Project/dataset/wikipedia_small/'#'/home/yash/Project/dataset/SoP_data/'
d= ""
thresh = 3

for file in files:
    d += open(path+file, 'r').read() # should be simple plain text file
    #data += f.split() #for word level encoding

data = preprocess(d)
chars = list(set(data))
#data = re.findall(r"\w+|[^\w\s]", d.lower(), re.UNICODE)
#chars = [k for k,v in Counter(data).items() if v>thresh]

UNK = len(chars)
#chars.append('UNKNOWN')

data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = 256 # size of hidden layer of neurons
seq_length = 16 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias



def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps, mask = {}, {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  p = 0.7
  # forward pass
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    
    # dropout mask
    mask[t] = (np.random.rand(hs[t].shape[0], hs[t].shape[1]) < p) / p 
    hs[t] *= mask[t] # drop!
    
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    #print(ps[t])
    #print(targets[t])
    #print(ps[t][targets[t],0])
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -10, 10, out=dparam) # clip to mitigate exploding gradients
  
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes



n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0


while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  
  inputs = [char_to_ix.get(ch, UNK) for ch in data[p:p+seq_length]]
  targets = [char_to_ix.get(ch, UNK) for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  if n % 1000 == 0:
    sample_ix = sample(hprev, inputs[0], 500)
    txt = ''.join(ix_to_char[ix]+'' for ix in sample_ix)
    print ('----\n %s \n----' % (txt, ))

  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print ('iter %d, loss: %f' % (n, smooth_loss)) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter 