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

def configure_session(gpu=None, intra_threads=None, inter_threads=None,
                      kmp_blocktime=None, kmp_affinity=None, omp_num_threads=None):
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

    if gpu is not None:
        gpu_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpu_devices[gpu], 'GPU')

    if intra_threads is not None:
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
    if inter_threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
