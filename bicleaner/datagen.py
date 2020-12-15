from keras.preprocessing.sequence import pad_sequences
from keras import layers, Model, models, optimizers
from keras import backend as K
import sentencepiece as sp
import numpy as np
import keras
import random
import sys

class TupleSentenceGenerator(keras.utils.Sequence):
    '''
    Generates batches of tuples of sentences and its labels if they have
    '''

    def __init__(self, sentencepiece_model,
            batch_size=64, maxlen=50, shuffle=False):
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.shuffle = shuffle
        self.num_samples = 0
        self.index = None
        self.x1 = None
        self.x2 = None
        self.y = None

        self.spm = sentencepiece_model


    def __len__(self):
        '''
        Length of epochs
        '''
        return int(np.ceil(self.x1.shape[0] / self.batch_size))

    #TODO investigate how to return batches reading from stdin
    def __getitem__(self, index):
        '''
        Return a batch of sentences
        '''
        # Avoid out of range when last batch smaller than batch_size
        if len(self)-1 == index:
            end = None
        else:
            end = (index+1)*self.batch_size
        start = index*self.batch_size
        indexes = self.index[start:end]

        return [ self.x1[indexes], self.x2[indexes] ], self.y[indexes]

    def on_epoch_end(self):
        'Shuffle indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.index)

    def load(self, source):
        '''
        Load sentences and encode to index numbers
        If source is a string it is considered a file,
        if it is a list is considered [text1_sentences, text2_sentences, tags]
        '''

        if isinstance(source, str):
            print('\tLoading data from:',source)
            data = [[], [], []]
            with open(source, 'r') as file_:
                for line in file_:
                    fields = line.split('\t')
                    data[0].append(fields[1].strip())
                    data[1].append(fields[2].strip())
                    data[2].append(fields[0])
        else:
            data = source

        self.x1 = pad_sequences(self.spm.encode(data[0]),
                                padding='post',
                                truncating='post',
                                maxlen=self.maxlen)
        self.x2 = pad_sequences(self.spm.encode(data[1]),
                                padding='post',
                                truncating='post',
                                maxlen=self.maxlen)
        self.num_samples = self.x1.shape[0]
        if data[2] is None: #TODO set y to None instead of zeros for inference
            self.y = np.zeros(self.num_samples)
        else:
            self.y = np.array(data[2], dtype=int)
        self.index = np.arange(0, self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.index) # Preventive shuffle in case data comes ordered
