"""CosmoFlow dataset specification"""

import os
import logging
from functools import partial

import numpy as np
import tensorflow as tf

def _parse_data(sample_proto, shape, apply_log=False):
    parsed_example = tf.io.parse_single_example(
        sample_proto,
        features = dict(x=tf.io.FixedLenFeature(shape, tf.float32),
                        y=tf.io.FixedLenFeature([4], tf.float32))
    )
    # Decode the data and normalize
    x, y = parsed_example['x'], parsed_example['y']
    if apply_log:
        # Trying logarithm of the data spectrum
        x = tf.math.log(x + tf.constant(1.))
    else:
        # Traditional mean normalization
        x /= (tf.reduce_sum(x) / np.prod(shape))
    return x, y

def construct_dataset(filenames, batch_size, n_epochs, sample_shape,
                      shard=0, n_shards=1, apply_log=False,
                      shuffle=False, shuffle_buffer_size=0,
                      prefetch=4):
    if len(filenames) == 0:
        return None

    # Define the dataset from the list of sharded, shuffled files
    data = tf.data.Dataset.from_tensor_slices(filenames)
    data = data.shard(num_shards=n_shards, index=shard)
    if shuffle:
        data = data.shuffle(len(filenames), reshuffle_each_iteration=True)

    # Parse TFRecords
    parse_data = partial(_parse_data, shape=sample_shape, apply_log=apply_log)
    data = data.apply(tf.data.TFRecordDataset).map(parse_data, num_parallel_calls=4)

    # Parallelize reading with interleave - no benefit?
    #data = data.interleave(
    #    lambda x: tf.data.TFRecordDataset(x).map(parse_data, num_parallel_calls=1),
    #    cycle_length=4
    #)

    # Localized sample shuffling (note: imperfect global shuffling).
    # Use if samples_per_file is greater than 1.
    if shuffle and shuffle_buffer_size > 0:
        data = data.shuffle(shuffle_buffer_size)

    # Construct batches
    data = data.repeat(n_epochs)
    data = data.batch(batch_size, drop_remainder=True)

    # Prefetch to device
    return data.prefetch(prefetch)

def get_datasets(data_dir, sample_shape, n_train, n_valid,
                 batch_size, n_epochs, dist=None, samples_per_file=1,
                 shuffle_train=True, shuffle_valid=False,
                 shard=True, staged_files=False,
                 prefetch=4, apply_log=False):

    # Determine number of staged file sets and worker shards
    n_file_sets = (dist.size // dist.local_size) if staged_files else 1
    if shard and staged_files:
        shard, n_shards = dist.local_rank, dist.local_size
    elif shard and not staged_files:
        shard, n_shards = dist.rank, dist.size
    else:
        shard, n_shards = 0, 1

    # Ensure samples divide evenly into files * local-disks * worker-shards * batches
    n_divs = samples_per_file * n_file_sets * n_shards * batch_size
    if (n_train % n_divs) != 0:
        logging.error('%i training samples not divisible by %i '
                      'samples_per_file * n_file_sets * n_shards * batch_size',
                      n_train, n_divs)
        raise Exception('Invalid sample counts')
    if (n_valid % n_divs) != 0:
        logging.error('%i validation samples not divisible by %i '
                      'samples_per_file * n_file_sets * n_shards * batch_size',
                      n_valid, n_divs)
        raise Exception('Invalid sample counts')
    n_train_files = n_train // (samples_per_file * n_file_sets)
    n_valid_files = n_valid // (samples_per_file * n_file_sets)
    n_train_steps = n_train // (n_file_sets * n_shards * batch_size)
    n_valid_steps = n_valid // (n_file_sets * n_shards * batch_size)

    if shard == 0:
        if staged_files:
            logging.info('Using %i locally-staged file sets', n_file_sets)
        logging.info('Splitting data into %i worker shards', n_shards)
        n_train_worker = n_train // (samples_per_file * n_file_sets * n_shards)
        n_valid_worker = n_valid // (samples_per_file * n_file_sets * n_shards)
        logging.info('Each worker reading %i training samples and %i validation samples',
                     n_train_worker, n_valid_worker)

    # Select the training and validation file lists
    data_dir = os.path.expandvars(data_dir)
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                        if f.endswith('.tfrecord')])
    train_files = all_files[:n_train_files]
    valid_files = all_files[n_train_files:n_train_files+n_valid_files]

    # Construct the data pipelines
    dataset_args = dict(sample_shape=sample_shape, batch_size=batch_size,
                        n_epochs=n_epochs, shard=shard, n_shards=n_shards,
                        prefetch=prefetch, apply_log=apply_log)
    train_dataset = construct_dataset(filenames=train_files,
                                      shuffle=shuffle_train,
                                      **dataset_args)
    valid_dataset = construct_dataset(filenames=valid_files,
                                      shuffle=shuffle_valid,
                                      **dataset_args)

    return dict(train_dataset=train_dataset, valid_dataset=valid_dataset,
                n_train_steps=n_train_steps, n_valid_steps=n_valid_steps)
