"""
Main training script for the CosmoFlow Keras benchmark
"""

# System imports
import os
import argparse
import logging

# External imports
import numpy as np
import tensorflow as tf
import yaml

# Local imports
from data import get_datasets
from models import get_model
from utils.optimizers import get_optimizer

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(logging.ERROR)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/cosmo.yaml')
    add_arg('-d', '--distributed', action='store_true')
    add_arg('-v', '--verbose', action='store_true')
    add_arg('--interactive', action='store_true')
    return parser.parse_args()

def config_logging(verbose):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)

def load_config(config_file):
    """Reads the YAML config file and returns a config dictionary"""
    with open(config_file) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def main():
    """Main function"""

    # Initialization
    args = parse_args()
    rank, n_ranks = 0, 1
    config = load_config(args.config)
    output_dir = os.path.expandvars(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    config_logging(verbose=args.verbose)
    logging.info('Initialized rank %i/%i', rank, n_ranks)
    if rank == 0:
        logging.info('Configuration: %s', config)

    # Load the data
    data_config = config['data']
    train_data, valid_data = get_datasets(**data_config)

    # Build the model
    model = get_model(**config['model'])

    # Configure the optimizer
    opt = get_optimizer(n_ranks=n_ranks,
                        distributed=args.distributed,
                        **config['optimizer'])

    # Compile the model
    train_config = config['train']
    model.compile(optimizer=opt, loss=train_config['loss'],
                  metrics=train_config['metrics'])
    if rank == 0:
        model.summary()

    # Prepare the callbacks
    callbacks = []

    if rank == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'checkpoint-{epoch:03d}.ckpt'),
            save_weights_only=True,
        ))

    # Train the model
    n_train = data_config['n_train_files'] * data_config['samples_per_file']
    n_valid = data_config['n_valid_files'] * data_config['samples_per_file']
    history = model.fit(train_data,
                        steps_per_epoch=n_train//data_config['batch_size'],
                        epochs=data_config['n_epochs'],
                        validation_data=valid_data,
                        validation_steps=n_valid//data_config['batch_size'],
                        callbacks=callbacks,
                        verbose=2)

    # Save the training history
    if rank == 0:
        # Print best metrics
        best_epoch = np.argmin(history.history['val_loss'])
        logging.info('Best epoch: %i', best_epoch)
        for key, val in history.history.items():
            logging.info('  %s: %.4f', key, val[best_epoch])

        # Save the history to file
        np.savez(os.path.join(output_dir, 'history'),
                 **history.history)

    if rank == 0:
        logging.info('All done!')

if __name__ == '__main__':
    main()
