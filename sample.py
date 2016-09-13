from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
#from six.moves import cPickle
import pickle
from utils import TextLoader
from model import Model

from six import text_type

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=5,
                       help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                       help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args)


def postprocess(string):
    
    #remove Initial spacing
    idx = 0
    while(string[idx] == ' '): idx += 1
    
    #capitalize First letters of sentences
    prv = string[idx]
    newstr = prv.upper()
    caps = False
    for pos, char in enumerate(string[idx+1:]):
        if prv == '.':
            caps = True
            
        if caps or ((char == 'i' or char == 'I') and (string[pos-1]==' ' and string[pos+1]==' ')):
            newstr += char.upper()
        else:
            newstr += char.lower()
        
        if newstr[-1].lower() != newstr[-1].upper():
            caps = False
            
        prv = char    
    return newstr
        
    
def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = pickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print(postprocess(model.sample(sess, chars, vocab, args.n, args.prime, args.sample)))

if __name__ == '__main__':
    main()
