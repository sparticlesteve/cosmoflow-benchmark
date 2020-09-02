"""
Hardware/device configuration
"""

# System
import os
import logging

# Externals
import tensorflow as tf

# Locals
import utils.distributed as dist

def configure_session(intra_threads=32, inter_threads=2,
                      kmp_blocktime=None, kmp_affinity=None, omp_num_threads=None,
                      gpu=None):
    """Sets the thread knobs in the TF backend"""
    if kmp_blocktime is not None:
        os.environ['KMP_BLOCKTIME'] = str(kmp_blocktime)
    if kmp_affinity is not None:
        os.environ['KMP_AFFINITY'] = kmp_affinity
    if omp_num_threads is not None:
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    if dist.rank() == 0:
        logging.info('KMP_BLOCKTIME %s', os.environ.get('KMP_BLOCKTIME', ''))
        logging.info('KMP_AFFINITY %s', os.environ.get('KMP_AFFINITY', ''))
        logging.info('OMP_NUM_THREADS %s', os.environ.get('OMP_NUM_THREADS', ''))
        logging.info('INTRA_THREADS %i', intra_threads)
        logging.info('INTER_THREADS %i', inter_threads)

    config = tf.ConfigProto(
        inter_op_parallelism_threads=inter_threads,
        intra_op_parallelism_threads=intra_threads
    )
    if gpu is not None:
        config.gpu_options.visible_device_list = str(gpu)
    tf.keras.backend.set_session(tf.Session(config=config))
