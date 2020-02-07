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
                      rank=0, n_ranks=1, shard=True, shuffle=False,                      
                      local_fs=False, shuffle_buffer_size=128):
    # Define the dataset from the list of files
    data = tf.data.Dataset.from_tensor_slices(filenames)
    if (shard and local_fs):
        local_rank = int(hvd.local_rank())
        local_size = int(hvd.local_size())
        data = data.shard(num_shards=local_size, index=local_rank)
    elif (shard):
        data = data.shard(num_shards=n_ranks, index=rank)
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

def get_datasets(data_dir, sample_shape, n_train_files, n_valid_files,
                 samples_per_file, batch_size, n_epochs,
                 shard=True, shuffle_train=True, shuffle_valid=False,
                 local_fs=False, rank=0, n_ranks=1):
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
                        if f.endswith('tfrecords')])
    train_files = all_files[:n_train_files]
    valid_files = all_files[n_train_files:n_train_files+n_valid_files]
    # Construct the data pipelines
    dataset_args = dict(sample_shape=sample_shape, batch_size=batch_size,
                        n_epochs=n_epochs, shard=shard, rank=rank, n_ranks=n_ranks, local_fs=local_fs)
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
