# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:38:09 2016

@author: yash
"""
import numpy as np
import re
import os

#a = "Aa. dd-l;s . AS .\"Hey!!\"."
class data:
    data = []
    files= []
    vectors = {}
    
    def __init__(self, folder, files = [], batch_size = 10):
        self.batch_size = batch_size
        self.folder = folder
        self.files = files
        self.counter = 0
        if len(files) == 0:
            self.files = [ file for file in os.listdir(folder) if file.split('.')[1] == 'txt']

        self.read_data()
        self.glove_embedding()

    def read_data(self):
        raw_data = []
        for file in self.files:
            path = self.folder+file
            #TODO check if data is concatenated as a SINGLE string
            raw_data += open(path, 'r').read()
        
        self.data = self.preprocess(raw_data)
        
        
    def preprocess(self, raw_data):
        #remove capital letter
        self.data = self.raw_data.lower()        
        #separate punctions and words, (words with underscore and hyphen are treated as single words)
        self.data = re.findall(r"\w+[-_]*\w+|[^\w\s]", self.data, re.UNICODE)
        #keep only unique words
        self.data_vocab = list(set(self.data))
        self.data_vocab_size = len(self.vocab)
        
    def get_next_batch(self, extra = 1):
        if (self.counter + self.batch_size + extra) > len(self.data):
            self.counter = 0
            print("Dataset over... Starting again.")
            
        d = []
        for item in self.data[self.counter: self.counter + self.batch_size + extra]:
            vec = self.vectors.get(item, [])            
            if len(vec) == 0:
                print("No Glove encoding for: "+item + "Using \'unk\' instead")
                d.append(self.vectors['unk'])
            else:
                d.append(vec)
                
        self.counter += self.batch_size
        
    def glove_embedding(self, prune = True):
#        with open(args.vocab_file, 'r') as f:
#            words = [x.rstrip().split(' ')[0] for x in f.readlines()]
        
        with open('/home/yash/Project/dataset/glove.6B/glove.6B.50d.txt', 'r') as f:
            for line in f:
                vals = line.rstrip().split(' ')
                if prune:
                    if vals[0] in self.data_vocab: #keep only the vectors of words in vocab
                            self.vectors[vals[0]] = [float(x) for x in vals[1:]]
                else:
                    self.vectors[vals[0]] = [float(x) for x in vals[1:]]
    
#        vocab_size = len(self.vectors.keys())
#        vocab  = {w: idx for idx, w in enumerate(words)}
#        ivocab = {idx: w for idx, w in enumerate(words)}
#    
#        vector_dim = len(self.vectors[ivocab[0]])
#        W = np.zeros((vocab_size, vector_dim))
#        for word, v in self.vectors.items():
#            if word == '<unk>':
#                continue
#            W[vocab[word], :] = v 
#            
#        # normalize each word vector to unit variance
#        W_norm = np.zeros(W.shape)
#        d = (np.sum(W ** 2, 1) ** (0.5))
#        W_norm = (W.T / d).T
#        
#        #scale to tanh range, i.e -1 to 1 
#        W_norm /= np.max(abs(W_norm))