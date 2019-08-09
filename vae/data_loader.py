# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import gzip
from keras.utils import Sequence



def get_generators(root, ratio, batch_size=1):
    filenames = os.listdir(root)
    t_indices, v_indices = validation_split(ratio, filenames)

    t = DataGenerator(root, filenames, t_indices, batch_size=batch_size)
    v = DataGenerator(root, filenames, v_indices, batch_size=batch_size)

    return t, v

def validation_split(ratio, filenames):
    nb_files = len(filenames)
    split_point = int(ratio * nb_files)

    indices = np.arange(nb_files)
    np.random.shuffle(indices)
    
    v_indices = indices[:split_point]
    t_indices = indices[split_point:]

    return t_indices, v_indices

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, root, filenames, indices, batch_size=1, shuffle=True):
        
        'Initialization'
        self.batch_size = batch_size
        self.filenames = filenames
        self.indices = indices
        self.shuffle = shuffle
        self.root = root
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        return self.__data_generation(self.indices[index*self.batch_size : (index+1)*self.batch_size])

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_data(self, idx):
        filename = self.filenames[idx]
        return np.load(self.root + filename)['bullet_emphasised']

    def __data_generation(self, subindices):
        'Generates data containing batch_size samples'
        X = self._load_data(subindices[0])

        for ID in subindices[1:]:
            X2 = self._load_data(ID)
            X = np.concatenate((X, X2))

        return X, X