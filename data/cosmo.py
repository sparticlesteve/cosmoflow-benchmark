"""CosmoFlow dataset specification"""

import os
import logging

import tensorflow as tf

def _parse_data(sample_proto):
    parsed_example = tf.parse_single_example(
        sample_proto,
        features = {'3Dmap': tf.FixedLenFeature([], tf.string),
                    'unitPar': tf.FixedLenFeature([], tf.string),
                    'physPar': tf.FixedLenFeature([], tf.string)}
    )
    # Decode the data and normalize
    data = tf.decode_raw(parsed_example['3Dmap'], tf.float32)    
    data = tf.reshape(data, [128, 128, 128, 1])
    data /= (tf.reduce_sum(data) / 128**3)
    # Decode the targets
    label = tf.decode_raw(parsed_example['unitPar'], tf.float32)
    return data, label

def construct_dataset(filenames, batch_size, n_epochs, shuffle=False,
                      rank=0, n_ranks=1, shuffle_buffer_size=128):
    # Define the TFRecord dataset
    data = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=4)
    # Shard into unique subsets for each worker
    data = data.apply(tf.data.experimental.filter_for_shard(
        num_shards=n_ranks, shard_index=rank))
    # Shuffle file list
    if shuffle:
        data = data.shuffle(len(filenames), reshuffle_each_iteration=True)
    # Parse samples out of the TFRecord files
    data = data.map(_parse_data)
    # Localized sample shuffling (note: imperfect global shuffling)
    if shuffle:
        data = data.shuffle(shuffle_buffer_size)
    data = data.repeat(n_epochs)
    data = data.batch(batch_size, drop_remainder=True)
    return data.prefetch(4)

def get_datasets(data_dir, n_train_files, n_valid_files,
                 samples_per_file, batch_size, n_epochs,
                 rank=0, n_ranks=1,
                 shuffle_train=True, shuffle_valid=False):
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
    train_dataset = construct_dataset(train_files, batch_size, n_epochs,
                                      rank=rank, n_ranks=n_ranks,
                                      shuffle=shuffle_train)
    valid_dataset = construct_dataset(valid_files, batch_size, n_epochs,
                                      rank=rank, n_ranks=n_ranks,
                                      shuffle=shuffle_valid)
    return train_dataset, valid_dataset
