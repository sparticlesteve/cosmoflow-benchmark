"""
This module contains some utility callbacks for Keras training.
"""

# System
from time import time

# Externals
import tensorflow as tf

class TimingCallback(tf.keras.callbacks.Callback):
    """A Keras Callback which records the time of each epoch"""
    def __init__(self):
        self.times = []
    
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time()

    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time() - self.starttime
        self.times.append(epoch_time)
