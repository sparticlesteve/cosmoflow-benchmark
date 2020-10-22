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

"""Generate training jobs for random HPO"""

import argparse

import numpy as np

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    add_arg('--n-evals', type=int, default=1,
            help='Number of HP points to evaluate')
    add_arg('--nodes', type=int, default=1,
            help='Number nodes per eval')
    add_arg('--config', default='configs/cosmo_hpo.yaml')
    #add_arg('--n-resume', type=int, default=1,
    #        help='Number of resuming submissions for each point')
    return parser.parse_args()

def main():
    """Main function"""

    # Parse command line arguments
    args = parse_args()

    # Sample the hyper-parameters
    hyper_params = [
        dict(
            apply_log = np.random.choice([1, 0]),
            conv_size = np.random.choice([8, 16, 32, 64]),
            fc1_size = np.random.choice([16, 32, 64, 128, 256, 512]),
            fc2_size = np.random.choice([16, 32, 64, 128, 256, 512]),
            dropout = np.random.random_sample() * 0.6,
            optimizer = np.random.choice(['Adam', 'Nadam']),
            lr = np.random.choice([5e-5, 1e-4, 5e-4, 1e-3, 5e-3]),
            hidden_activation = np.random.choice(['ReLU', 'LeakyReLU']),
        ) for i in range(args.n_evals)
    ]

    # Generate the evaluation commands
    cmd = ('sbatch -N {nodes} -J cosmo-hpo scripts/train_cgpu_requeue.sh'
           ' {config} --apply-log {apply_log} --conv-size {conv_size}'
           ' --fc1-size {fc1_size} --fc2-size {fc2_size}'
           ' --dropout {dropout} --optimizer {optimizer}'
           ' --lr {lr} --hidden-activation {hidden_activation}')
    eval_commands = [cmd.format(nodes=args.nodes, config=args.config, **hp)
                     for hp in hyper_params]

    for eval_command in eval_commands:
        print(eval_command)

if __name__ == '__main__':
    main()
