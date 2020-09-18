"""
This module contains some utility callbacks for Keras training.
"""

# System
from time import time
import logging

# Externals
import tensorflow as tf
from mlperf_logging import mllog

class MLPerfLoggingCallback(tf.keras.callbacks.Callback):
    """A Keras Callback for logging MLPerf results"""
    def __init__(self, metric='val_mean_absolute_error', log_key='eval_error'):
        self.mllogger = mllog.get_mllogger()
        self.metric = metric
        self.log_key = log_key

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
        eval_metric = logs[self.metric]
        self.mllogger.event(key=self.log_key, value=eval_metric,
                            metadata={'epoch_num': epoch})

class StopAtTargetCallback(tf.keras.callbacks.Callback):
    """A Keras callback for stopping training at specified target quality"""

    def __init__(self, metric='val_mean_absolute_error', target_max=None):
        self.metric = metric
        self.target_max = target_max

    def on_epoch_end(self, epoch, logs={}):
        eval_metric = logs[self.metric]
        if self.target_max is not None and eval_metric <= self.target_max:
            self.model.stop_training = True
            logging.info('Target reached; stopping training')

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
