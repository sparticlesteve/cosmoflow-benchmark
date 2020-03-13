"""Data loading benchmark code for cosmoflow-benchmark

This script can be used to test just the data-loading part of the CosmoFlow
application to understand I/O performance.
"""

# System imports
import argparse
import time
import pprint
from types import SimpleNamespace

# External imports
import tensorflow as tf

# Local imports
from data import get_datasets

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--data-dir', default='/global/cscratch1/sd/sfarrell/cosmoflow-benchmark/data/cosmoUniverse_2019_05_4parE_tf')
    add_arg('--n-samples', type=int, default=512)
    add_arg('--batch-size', type=int, default=4)
    add_arg('--n-epochs', type=int, default=1)
    add_arg('--inter-threads', type=int, default=2)
    add_arg('--intra-threads', type=int, default=32)
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Session setup
    tf.compat.v1.enable_eager_execution(
        config=tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=args.inter_threads,
            intra_op_parallelism_threads=args.intra_threads))

    # Not running distributed
    dist = SimpleNamespace(rank=0, size=1, local_rank=0, local_size=1)

    # Load the dataset
    data = get_datasets(name='cosmo',
                        data_dir=args.data_dir,
                        sample_shape=[128, 128, 128, 4],
                        n_train=args.n_samples,
                        n_valid=0,
                        batch_size=args.batch_size,
                        n_epochs=args.n_epochs,
                        apply_log=True,
                        shard=False,
                        dist=dist)

    pprint.pprint(data)

    start_time = time.perf_counter()
    for x, y in data['train_dataset']:
        # Perform a simple operation
        tf.math.reduce_sum(x)
        tf.math.reduce_sum(y)
    duration = time.perf_counter() - start_time

    print('Total time: %.4f s' % duration)
    print('Throughput: %.4f samples/s' % (args.n_samples / duration))

    print('All done!')

if __name__ == '__main__':
    main()
