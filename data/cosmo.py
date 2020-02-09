"""CosmoFlow dataset specification"""

import os
import logging
from functools import partial

import numpy as np
import tensorflow as tf

import horovod.tensorflow.keras as hvd

def _parse_data(sample_proto, shape):
    parsed_example = tf.parse_single_example(
        sample_proto,
        features = {'3Dmap': tf.FixedLenFeature([], tf.string),
                    'unitPar': tf.FixedLenFeature([], tf.string),
                    'physPar': tf.FixedLenFeature([], tf.string)}
    )
    # Decode the data and normalize
    data = tf.decode_raw(parsed_example['3Dmap'], tf.float32)    
    data = tf.reshape(data, shape)
    data /= (tf.reduce_sum(data) / np.prod(shape))
    # Decode the targets
    label = tf.decode_raw(parsed_example['unitPar'], tf.float32)
    return data, label

def construct_dataset(filenames, batch_size, n_epochs, sample_shape,
                      shard=0, n_shards=1, shuffle=False,
                      shuffle_buffer_size=128):
    if len(filenames) == 0:
        return None
    # Define the dataset from the list of files
    data = tf.data.Dataset.from_tensor_slices(filenames)
    data = data.shard(num_shards=n_shards, index=shard)
    if shuffle:
        data = data.shuffle(len(filenames), reshuffle_each_iteration=True)
    # Parse TFRecords
    parse_data = partial(_parse_data, shape=sample_shape)
    data = data.apply(tf.data.TFRecordDataset).map(parse_data, num_parallel_calls=4)
    # Localized sample shuffling (note: imperfect global shuffling)
    if shuffle:
        data = data.shuffle(shuffle_buffer_size)
    data = data.repeat(n_epochs)
    data = data.batch(batch_size, drop_remainder=True)
    return data.prefetch(4)

def get_datasets(dist, data_dir, sample_shape,
                 n_train, n_valid,
                 samples_per_file, batch_size, n_epochs,
                 shuffle_train=True, shuffle_valid=False,
                 shard_type='global'):

    # Ensure sample counts divide into files
    if (n_train % samples_per_file) != 0:
        raise Exception('n_train (%i) must be divisible by %i samples/file' %
                        (n_train, samples_per_file))
    if (n_valid % samples_per_file) != 0:
        raise Exception('n_valid (%i) must be divisible by %i samples/file' %
                        (n_valid, samples_per_file))
    n_train_files = n_train // samples_per_file
    n_valid_files = n_valid // samples_per_file

    # Determine number of data shards
    if shard_type == 'global':
        shard, n_shards = dist.rank, dist.size
    elif shard_type == 'local':
        shard, n_shards = dist.local_rank, dist.local_size
    else:
        shard, n_shards = 0, 1

    # Ensure file counts divide evenly into worker shards
    if (n_train_files % n_shards) != 0:
        raise Exception('n_train_files (%i) must be divisible by %i shards' %
                        (n_train_files, n_shards))
    if (n_valid_files % n_shards) != 0:
        raise Exception('n_valid_files (%i) must be divisible by %i shards' %
                        (n_valid_files, n_shards))

    n_train_steps = n_train // n_shards // batch_size
    n_valid_steps = n_valid // n_shards // batch_size
    if shard == 0:
        logging.info('Loading %i training samples from %i files', n_train, n_train_files)
        logging.info('Loading %i validation samples from %i files', n_valid, n_valid_files)
        logging.info('Splitting data into %i shard(s)', n_shards)

    # Select the training and validation file lists
    data_dir = os.path.expandvars(data_dir)
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                        if f.endswith('tfrecords')])
    train_files = all_files[:n_train_files]
    valid_files = all_files[n_train_files:n_train_files+n_valid_files]

    # Construct the data pipelines
    dataset_args = dict(sample_shape=sample_shape, batch_size=batch_size,
                        n_epochs=n_epochs, shard=shard, n_shards=n_shards)
    train_dataset = construct_dataset(filenames=train_files,
                                      shuffle=shuffle_train,
                                      **dataset_args)
    valid_dataset = construct_dataset(filenames=valid_files,
                                      shuffle=shuffle_valid,
                                      **dataset_args)

    return dict(train_dataset=train_dataset, valid_dataset=valid_dataset,
                n_train=n_train, n_valid=n_valid, n_train_steps=n_train_steps,
                n_valid_steps=n_valid_steps)
