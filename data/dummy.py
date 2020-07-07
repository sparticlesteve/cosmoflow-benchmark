"""
Random dummy dataset specification.
"""

# System
import math

# Externals
import tensorflow as tf


def construct_dataset(sample_shape, target_shape,
                       batch_size=1, n_samples=32):
    x = tf.random.uniform([n_samples]+sample_shape)
    y = tf.random.uniform([n_samples]+target_shape)
    data = tf.data.Dataset.from_tensor_slices((x, y))
    return data.repeat().batch(batch_size).prefetch(4)

def get_datasets(sample_shape, target_shape, batch_size,
                 n_train, n_valid, dist, n_epochs=None, shard=False):
    train_dataset = construct_dataset(sample_shape, target_shape, batch_size=batch_size)
    valid_dataset = None
    if n_valid > 0:
        valid_dataset = construct_dataset(sample_shape, target_shape, batch_size=batch_size)
    n_train_steps = n_train  // batch_size
    n_valid_steps = n_valid  // batch_size
    if shard:
        n_train_steps = n_train_steps // dist.size
        n_valid_steps = n_valid_steps // dist.size

    return dict(train_dataset=train_dataset, valid_dataset=valid_dataset,
                n_train=n_train, n_valid=n_valid, n_train_steps=n_train_steps,
                n_valid_steps=n_valid_steps)
