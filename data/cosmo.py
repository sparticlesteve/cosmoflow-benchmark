"""CosmoFlow dataset specification"""

import os
import logging
from functools import partial

import numpy as np
import tensorflow as tf

def _parse_data(sample_proto, shape):
    parsed_example = tf.parse_single_example(
        sample_proto,
        features = dict(x=tf.FixedLenFeature(shape, tf.float32),
                        y=tf.FixedLenFeature([4], tf.float32))
    )
    # Decode the data and normalize
    x, y = parsed_example['x'], parsed_example['y']
    x /= (tf.reduce_sum(x) / np.prod(shape))
    return x, y

def construct_dataset(filenames, batch_size, n_epochs, sample_shape,
                      rank=0, n_ranks=1, shard=True, shuffle=False):

    # Define the dataset from the list of files
    data = tf.data.Dataset.from_tensor_slices(filenames)
    if shard:
        data = data.shard(num_shards=n_ranks, index=rank)
    if shuffle:
        data = data.shuffle(len(filenames), reshuffle_each_iteration=True)
    # Parse TFRecords
    parse_data = partial(_parse_data, shape=sample_shape)
    data = data.apply(tf.data.TFRecordDataset).map(parse_data, num_parallel_calls=4)
    data = data.repeat(n_epochs)
    data = data.batch(batch_size, drop_remainder=True)
    return data.prefetch(4)

def get_datasets(data_dir, sample_shape, n_train_files, n_valid_files,
                 batch_size, n_epochs, samples_per_file=1,
                 shard=True, shuffle_train=True, shuffle_valid=False,
                 rank=0, n_ranks=1):
    # Ensure file counts divide evenly into worker shards
    if shard:
        n_train_files = (n_train_files // n_ranks) * n_ranks
        n_valid_files = (n_valid_files // n_ranks) * n_ranks
    n_train = n_train_files * samples_per_file
    n_valid = n_valid_files * samples_per_file
    n_train_steps = n_train // batch_size
    n_valid_steps = n_valid // batch_size
    if shard:
        n_train_steps = n_train_steps // n_ranks
        n_valid_steps = n_valid_steps // n_ranks
    if rank == 0:
        logging.info('Loading %i training samples from %i files', n_train, n_train_files)
        logging.info('Loading %i validation samples from %i files', n_valid, n_valid_files)
    # Select the training and validation file lists
    data_dir = os.path.expandvars(data_dir)
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                        if f.endswith('.tfrecord')])
    train_files = all_files[:n_train_files]
    valid_files = all_files[n_train_files:n_train_files+n_valid_files]
    # Construct the data pipelines
    dataset_args = dict(sample_shape=sample_shape, batch_size=batch_size,
                        n_epochs=n_epochs, shard=shard, rank=rank, n_ranks=n_ranks)
    train_dataset = construct_dataset(filenames=train_files,
                                      shuffle=shuffle_train,
                                      **dataset_args)
    if n_valid > 0:
        valid_dataset = construct_dataset(filenames=valid_files,
                                          shuffle=shuffle_valid,
                                          **dataset_args)
    else:
        valid_dataset = None

    return dict(train_dataset=train_dataset, valid_dataset=valid_dataset,
                n_train=n_train, n_valid=n_valid, n_train_steps=n_train_steps,
                n_valid_steps=n_valid_steps)
