"""Utility code for handling checkpoint loading"""

# System imports
import os
import logging

# External imports
import tensorflow as tf
import horovod.tensorflow.keras as hvd

def reload_last_checkpoint(checkpoint_format, n_epochs, distributed):
    """Finds and loads the last checkpoint matching the provided pattern"""
    # Count down from n_epochs to 0 to find the last epoch.
    # Note that keras names checkpoint files with epoch number starting from 1.
    # So the matched number corresponds to the new initial epoch.
    for epoch in range(n_epochs, 0, -1):
        checkpoint = checkpoint_format.format(epoch=epoch)
        if os.path.exists(checkpoint):
            logging.info('Found last checkpoint at %s', checkpoint)
            # Use horovod's reload to prepare the DistributedOptimizer
            if distributed:
                model = hvd.load_model(checkpoint)
            else:
                model = tf.keras.models.load_model(checkpoint)
            return epoch, model
    raise Exception('Unable to find a checkpoint file at %s' % checkpoint_format)
