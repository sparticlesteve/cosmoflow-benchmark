"""Utility code for handling checkpoint loading"""

# System imports
import os
import logging

# External imports
import tensorflow as tf
import horovod.tensorflow.keras as hvd

def load_hvd_model(checkpoint):
    """Load model with Horovod setup.

    This exists as a workaround for my horovod issue:
    https://github.com/horovod/horovod/issues/1920

    It takes care of wrapping the checkpoint optimizer in the horovod
    DistributedOptimizer.

    I've dropped support for compression, here, which may be useful.

    See:
    https://github.com/horovod/horovod/blob/master/horovod/tensorflow/keras/__init__.py
    https://github.com/horovod/horovod/blob/master/horovod/_keras/__init__.py
    """
    def wrap_optimizer(cls):
        return lambda **kwargs: hvd.DistributedOptimizer(cls(**kwargs))
    horovod_objects = {
        subclass.__name__.lower(): wrap_optimizer(subclass)
        for subclass in tf.keras.optimizers.Optimizer.__subclasses__()
        # This is the line that doesn't work in issue horovod/1920
        #if subclass.__module__ == keras.optimizers.Optimizer.__module__
    }
    return tf.keras.models.load_model(checkpoint, custom_objects=horovod_objects)

def reload_last_checkpoint(checkpoint_format, n_epochs, distributed):
    """Finds and loads the last checkpoint matching the provided pattern"""
    # Count down from n_epochs to 0 to find the last epoch.
    # Note that keras names checkpoint files with epoch number starting from 1.
    # So the matched number corresponds to the new initial epoch.
    for epoch in range(n_epochs, 0, -1):
        checkpoint = checkpoint_format.format(epoch=epoch)
        if os.path.exists(checkpoint):
            logging.info('Found last checkpoint at %s', checkpoint)
            # Use special reload to prepare the DistributedOptimizer
            if distributed:
                model = load_hvd_model(checkpoint)
            else:
                model = tf.keras.models.load_model(checkpoint)
            return epoch, model
    raise Exception('Unable to find a checkpoint file at %s' % checkpoint_format)
