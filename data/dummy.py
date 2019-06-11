"""
Random dummy dataset specification.
"""

# System
import math

# Externals
import numpy as np
from keras.utils import Sequence

class DummyDataset(Sequence):

    def __init__(self, n_samples, batch_size, input_shape, target_shape):
        self.x = np.random.normal(size=(n_samples,) + tuple(input_shape))
        self.y = np.random.normal(size=(n_samples,) + tuple(target_shape))
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        return self.x[start:end], self.y[start:end]

def get_datasets(batch_size, n_train=1024, n_valid=1024,
                 input_shape=(32, 32, 3), target_shape=()):
    train_data = DummyDataset(n_train, batch_size, input_shape, target_shape)
    valid_data = DummyDataset(n_valid, batch_size, input_shape, target_shape)
    return train_data, valid_data
