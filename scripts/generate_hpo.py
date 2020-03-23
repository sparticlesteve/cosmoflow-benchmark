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
