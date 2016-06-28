# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:38:09 2016

@author: yash
"""
import numpy as np
import re
import os

#a = "Aa. dd-l;s . AS .\"Hey!!\"."
class Data:
    data = []
    files= []
    vectors = {}
    has_more_data = True
    
    def __init__(self, folder, files = [], batch_size = 10, seq_len = 25):
        self.batch_size = batch_size
        self.folder = folder
        self.files = files
        self.seq_len = seq_len
        self.counter = 0
        if len(files) == 0:
            self.files = [ file for file in os.listdir(folder) if file.split('.')[1] == 'txt']

        self.read_data()
        print("Done reading data")
        print("length of data: ", len(self.data))
        print("length of vocab: ", self.data_vocab_size)
        self.glove_embedding()
        print("Done glove embedding")

    def read_data(self):
        raw_data = ''
        for file in self.files:
            path = self.folder+file
            #TODO check if data is concatenated as a SINGLE string
            raw_data += open(path, 'r').read()
        
        self.data = self.preprocess(raw_data)
        
        
    def preprocess(self, raw_data):
        #remove capital letter       
        #separate punctions and words, (words with underscore and hyphen are treated as single words)
        processed = re.findall(r"\w+|[^\w\s]", raw_data.lower(), re.UNICODE)
        #keep only unique words
        self.data_vocab = list(set(processed))
        self.data_vocab_size = len(self.data_vocab)
        
        return processed
        
    def get_next_batch(self, extra = 1):            
        d_batch = []
        for bat in range(self.batch_size):
            d_seq = []
            for item in self.data[self.counter: self.counter + self.seq_len + extra]:
                vec = self.vectors.get(item, [])            
                if len(vec) == 0:
                    #print("No Glove encoding for: "+item + " Using \'unk\' instead")
                    d_seq.append(self.vectors['unk'])
                    #print(self.vectors.get('unk', []) )
                else:
                    d_seq.append(vec)
                    
            self.counter += self.seq_len
            
            if (self.counter + self.seq_len + extra) > len(self.data):
                self.counter = 0
                self.has_more_data = False
                print("Dataset over... Starting afresh.") 

           
            d_batch.append(d_seq)
        return np.array(d_batch)
        
    def glove_embedding(self, prune = True):
#        with open(args.vocab_file, 'r') as f:
#            words = [x.rstrip().split(' ')[0] for x in f.readlines()]
        
        with open('/home/yash/Project/dataset/glove.6B/glove.6B.50d.txt', 'r') as f:
            for line in f:
                vals = line.rstrip().split(' ')
                if prune:
                    if vals[0] in self.data_vocab or vals[0]=='unk': #keep only the vectors of words in vocab
                            self.vectors[vals[0]] = [float(x) for x in vals[1:]]
                else:
                    self.vectors[vals[0]] = [float(x) for x in vals[1:]]
        
        #scale the vectors b/w [-1,1]            
        maxi = np.max(list(self.vectors.values()))
        mini = np.min(list(self.vectors.values()))
        if abs(mini) > abs(maxi):
            maxi = abs(mini)
            
        for key, val in self.vectors.items():
            self.vectors[key] = [v/maxi for v in val]
    
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