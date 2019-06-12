"""
Main training script for the CosmoFlow Keras benchmark
"""

# System imports
import os
import argparse
import logging

# External imports
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf

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
    add_arg('--resume', action='store_true',
            help='Resume from last checkpoint')
    return parser.parse_args()

def config_logging(verbose):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)

def load_config(config_file):
    """Reads the YAML config file and returns a config dictionary"""
    with open(config_file) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_history(output_dir):
    return pd.read_csv(os.path.join(output_dir, 'history.csv'))

def print_training_summary(output_dir):
    history = load_history(output_dir)
    best = history.val_loss.idxmin()
    logging.info('Best result:')
    for key in history.keys():
        logging.info('  %s: %g', key, history[key].loc[best])

def reload_last_checkpoint(checkpoint_format, n_epochs):
    for epoch in range(n_epochs, 0, -1):
        checkpoint = checkpoint_format.format(epoch=epoch)
        if os.path.exists(checkpoint):
            model = tf.keras.models.load_model(checkpoint)
            return epoch, model
    raise Exception('Unable to find a checkpoint file at %s' % checkpoint_format)

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

    # Construct or reload the model
    initial_epoch = 0
    checkpoint_format = os.path.join(output_dir, 'checkpoint-{epoch:03d}.h5')
    if args.resume:
        # Reload model from last checkpoint
        initial_epoch, model = reload_last_checkpoint(
            checkpoint_format, data_config['n_epochs'])
    else:
        # Build a new model
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
        # Checkpointing
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_format))
        # CSV logging
        callbacks.append(tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'history.csv'), append=args.resume))

    # Train the model
    n_train = data_config['n_train_files'] * data_config['samples_per_file']
    n_valid = data_config['n_valid_files'] * data_config['samples_per_file']
    model.fit(train_data,
              steps_per_epoch=n_train//data_config['batch_size'],
              epochs=data_config['n_epochs'],
              validation_data=valid_data,
              validation_steps=n_valid//data_config['batch_size'],
              callbacks=callbacks,
              initial_epoch=initial_epoch,
              verbose=(1 if args.verbose else 2))

    # Print training summary
    if rank == 0:
        print_training_summary(output_dir)

    if rank == 0:
        logging.info('All done!')

if __name__ == '__main__':
    main()
