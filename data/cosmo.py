"""CosmoFlow dataset specification"""

import os
import logging
from functools import partial

import numpy as np

import tensorflow as tf

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
                      rank=0, n_ranks=1, shuffle=False,
                      shuffle_buffer_size=128):
    # Define the TFRecord dataset
    data = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=4)
    # Shard into unique subsets for each worker
    data = data.apply(tf.data.experimental.filter_for_shard(
        num_shards=n_ranks, shard_index=rank))
    # Shuffle file list
    if shuffle:
        data = data.shuffle(len(filenames), reshuffle_each_iteration=True)
    # Parse samples out of the TFRecord files
    parse_data = partial(_parse_data, shape=sample_shape)
    data = data.map(parse_data)
    # Localized sample shuffling (note: imperfect global shuffling)
    if shuffle:
        data = data.shuffle(shuffle_buffer_size)
    data = data.repeat(n_epochs)
    data = data.batch(batch_size, drop_remainder=True)
    return data.prefetch(4)

def get_datasets(data_dir, sample_shape, n_train_files, n_valid_files,
                 samples_per_file, batch_size, n_epochs,
                 shuffle_train=True, shuffle_valid=False,
                 rank=0, n_ranks=1):
    # Ensure file counts divide evenly into worker shards
    n_train_files = (n_train_files // n_ranks) * n_ranks
    n_valid_files = (n_valid_files // n_ranks) * n_ranks
    logging.info('Loading %i training samples from %i files',
                 n_train_files * samples_per_file, n_train_files)
    logging.info('Loading %i validation samples from %i files',
                 n_valid_files * samples_per_file, n_valid_files)
    # Select the training and validation file lists
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                 if f.endswith('tfrecords')]
    train_files = all_files[:n_train_files]
    valid_files = all_files[n_train_files:n_train_files+n_valid_files]
    # Construct the data pipelines
    dataset_args = dict(sample_shape=sample_shape, batch_size=batch_size,
                        n_epochs=n_epochs, rank=rank, n_ranks=n_ranks)
    train_dataset = construct_dataset(filenames=train_files,
                                      shuffle=shuffle_train,
                                      **dataset_args)
    valid_dataset = construct_dataset(filenames=valid_files,
                                      shuffle=shuffle_valid,
                                      **dataset_args)
    return train_dataset, valid_dataset
