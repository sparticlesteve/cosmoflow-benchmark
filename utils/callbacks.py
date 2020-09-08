"""
This module contains some utility callbacks for Keras training.
"""

# System
from time import time

# Externals
import tensorflow as tf
from mlperf_logging import mllog

class MLPerfLoggingCallback(tf.keras.callbacks.Callback):
    """A Keras Callback for logging MLPerf results"""
    def __init__(self):
        self.mllogger = mllog.get_mllogger()

    def on_epoch_begin(self, epoch, logs={}):
        self.mllogger.start(key=mllog.constants.EPOCH_START,
                            metadata={'epoch_num': epoch})
        self._epoch = epoch

    def on_test_begin(self, logs):
        self.mllogger.start(key=mllog.constants.EVAL_START,
                            metadata={'epoch_num': self._epoch})

    def on_test_end(self, logs):
        self.mllogger.end(key=mllog.constants.EVAL_STOP,
                          metadata={'epoch_num': self._epoch})

    def on_epoch_end(self, epoch, logs={}):
        self.mllogger.end(key=mllog.constants.EPOCH_STOP,
                          metadata={'epoch_num': epoch})
        val_mae = logs['val_mean_absolute_error']
        self.mllogger.event(key='eval_error', value=val_mae,
                            metadata={'epoch_num': epoch})

class TimingCallback(tf.keras.callbacks.Callback):
    """A Keras Callback which records the time of each epoch"""
    def __init__(self):
        self.times = []
    
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time()

    def on_epoch_end(self, epoch, logs={}):
        epoch_time = time() - self.starttime
        self.times.append(epoch_time)
        logs['time'] = epoch_time

#class LearningRateScheduleCallback(tf.keras.callbacks.Callback):
#    def __init__(self, multiplier,
#                 start_epoch=0, end_epoch=None,
#                 momentum_correction=True):
#        super().__init__()
#        self.start_epoch = start_epoch
#        self.end_epoch = end_epoch
#        self.momentum_correction = momentum_correction
#        self.initial_lr = None
#        self.restore_momentum = None
