# 'Regression of 3D Sky Map to Cosmological Parameters (CosmoFlow)'
# Copyright (c) 2018, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S. Dept. of Energy).  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Innovation & Partnerships Office at IPO@lbl.gov.
#
# NOTICE.  This Software was developed under funding from the U.S. Department of
# Energy and the U.S. Government consequently retains certain rights. As such,
# the U.S. Government has been granted for itself and others acting on its
# behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
# to reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit other to do so.

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
