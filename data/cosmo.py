"""CosmoFlow dataset specification"""

# System imports
import os
import logging
import glob
from functools import partial

# External imports
import numpy as np
import tensorflow as tf
from mlperf_logging import mllog
import horovod.tensorflow.keras as hvd

# Local imports
from utils.staging import stage_files

def _parse_data(sample_proto, shape, apply_log=False):
    """Parse the data out of the TFRecord proto buf.

    This pipeline could be sped up considerably by moving the cast and log
    transform onto the GPU, in the model (e.g. in a keras Lambda layer).
    """

    # Parse the serialized features
    feature_spec = dict(x=tf.io.FixedLenFeature([], tf.string),
                        y=tf.io.FixedLenFeature([4], tf.float32))
    parsed_example = tf.io.parse_single_example(
        sample_proto, features=feature_spec)

    # Decode the bytes data, convert to float
    x = tf.decode_raw(parsed_example['x'], tf.int16)
    x = tf.cast(tf.reshape(x, shape), tf.float32)
    y = parsed_example['y']

    # Data normalization/scaling
    if apply_log:
        # Take logarithm of the data spectrum
        x = tf.math.log(x + tf.constant(1.))
    else:
        # Traditional mean normalization
        x /= (tf.reduce_sum(x) / np.prod(shape))

    return x, y

def construct_dataset(file_dir, n_samples, batch_size, n_epochs,
                      sample_shape, samples_per_file=1, n_file_sets=1,
                      shard=0, n_shards=1, apply_log=False,
                      randomize_files=False, shuffle=False,
                      shuffle_buffer_size=0, n_parallel_reads=4, prefetch=4):
    """This function takes a folder with files and builds the TF dataset.

    It ensures that the requested sample counts are divisible by files,
    local-disks, worker shards, and mini-batches.
    """

    if n_samples == 0:
        return None, 0

    # Ensure samples divide evenly into files * local-disks * worker-shards * batches
    n_divs = samples_per_file * n_file_sets * n_shards * batch_size
    if (n_samples % n_divs) != 0:
        logging.error('Number of samples (%i) not divisible by %i '
                      'samples_per_file * n_file_sets * n_shards * batch_size',
                      n_samples, n_divs)
        raise Exception('Invalid sample counts')

    # Number of files and steps
    n_files = n_samples // (samples_per_file * n_file_sets)
    n_steps = n_samples // (n_file_sets * n_shards * batch_size)

    # Find the files
    filenames = sorted(glob.glob(os.path.join(file_dir, '*.tfrecord')))
    assert (0 <= n_files) and (n_files <= len(filenames)), (
        'Requested %i files, %i available' % (n_files, len(filenames)))
    if randomize_files:
        np.random.shuffle(filenames)
    filenames = filenames[:n_files]

    # Define the dataset from the list of sharded, shuffled files
    data = tf.data.Dataset.from_tensor_slices(filenames)
    data = data.shard(num_shards=n_shards, index=shard)
    if shuffle:
        data = data.shuffle(len(filenames), reshuffle_each_iteration=True)

    # Parse TFRecords
    parse_data = partial(_parse_data, shape=sample_shape, apply_log=apply_log)
    data = data.apply(tf.data.TFRecordDataset).map(
        parse_data, num_parallel_calls=n_parallel_reads)

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
    return data.prefetch(prefetch), n_steps

def get_datasets(data_dir, sample_shape, n_train, n_valid,
                 batch_size, n_epochs, dist, samples_per_file=1,
                 shuffle_train=True, shuffle_valid=False,
                 shard=True, stage_dir=None, apply_log=False,
                 **kwargs):
    """Prepare TF datasets for training and validation.

    This function will perform optional staging of data chunks to local
    filesystems. It also figures out how to split files according to local
    filesystems (if pre-staging) and worker shards (if sharding).

    Returns: A dict of the two datasets and step counts per epoch.
    """

    # MLPerf logging
    if dist.rank == 0:
        mllogger = mllog.get_mllogger()
        mllogger.event(key=mllog.constants.GLOBAL_BATCH_SIZE, value=batch_size*dist.size)
        mllogger.event(key=mllog.constants.TRAIN_SAMPLES, value=n_train)
        mllogger.event(key=mllog.constants.EVAL_SAMPLES, value=n_valid)
    data_dir = os.path.expandvars(data_dir)

    # Local data staging
    if stage_dir is not None:
        staged_files = True
        # Stage training data
        stage_files(os.path.join(data_dir, 'train'),
                    os.path.join(stage_dir, 'train'),
                    n_files=n_train, rank=dist.rank, size=dist.size)
        # Stage validation data
        stage_files(os.path.join(data_dir, 'validation'),
                    os.path.join(stage_dir, 'validation'),
                    n_files=n_valid, rank=dist.rank, size=dist.size)
        data_dir = stage_dir

        # Barrier to ensure all workers are done transferring
        if dist.size > 0:
            hvd.allreduce([], name="Barrier")
    else:
        staged_files = False

    # Determine number of staged file sets and worker shards
    n_file_sets = (dist.size // dist.local_size) if staged_files else 1
    if shard and staged_files:
        shard, n_shards = dist.local_rank, dist.local_size
    elif shard and not staged_files:
        shard, n_shards = dist.rank, dist.size
    else:
        shard, n_shards = 0, 1

    # Construct the training and validation datasets
    dataset_args = dict(batch_size=batch_size, n_epochs=n_epochs,
                        sample_shape=sample_shape, samples_per_file=samples_per_file,
                        n_file_sets=n_file_sets, shard=shard, n_shards=n_shards,
                        apply_log=apply_log, **kwargs)
    train_dataset, n_train_steps = construct_dataset(
        file_dir=os.path.join(data_dir, 'train'),
        n_samples=n_train, shuffle=shuffle_train, **dataset_args)
    valid_dataset, n_valid_steps = construct_dataset(
        file_dir=os.path.join(data_dir, 'validation'),
        n_samples=n_valid, shuffle=shuffle_valid, **dataset_args)

    if shard == 0:
        if staged_files:
            logging.info('Using %i locally-staged file sets', n_file_sets)
        logging.info('Splitting data into %i worker shards', n_shards)
        n_train_worker = n_train // (samples_per_file * n_file_sets * n_shards)
        n_valid_worker = n_valid // (samples_per_file * n_file_sets * n_shards)
        logging.info('Each worker reading %i training samples and %i validation samples',
                     n_train_worker, n_valid_worker)

    if dist.rank == 0:
        logging.info('Data setting n_train: %i', n_train)
        logging.info('Data setting n_valid: %i', n_valid)
        logging.info('Data setting batch_size: %i', batch_size)
        for k, v in kwargs.items():
            logging.info('Data setting %s: %s', k, v)

    return dict(train_dataset=train_dataset, valid_dataset=valid_dataset,
                n_train_steps=n_train_steps, n_valid_steps=n_valid_steps)
