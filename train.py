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
import horovod.tensorflow.keras as hvd

# Local imports
from data import get_datasets
from models import get_model
from utils.optimizers import get_optimizer
from utils.callbacks import TimingCallback
from utils.device import configure_session

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(logging.ERROR)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/cosmo.yaml')
    add_arg('-d', '--distributed', action='store_true')
    add_arg('--rank-gpu', action='store_true',
            help='Use GPU based on local rank')
    add_arg('--resume', action='store_true',
            help='Resume from last checkpoint')
    add_arg('-v', '--verbose', action='store_true')
    return parser.parse_args()

def init_workers(distributed=False):
    rank, local_rank, n_ranks = 0, 0, 1
    if distributed:
        hvd.init()
        rank, local_rank, n_ranks = hvd.rank(), hvd.local_rank(), hvd.size()
    return rank, local_rank, n_ranks

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
    if 'val_loss' in history.keys():
        best = history.val_loss.idxmin()
        logging.info('Best result:')
        for key in history.keys():
            logging.info('  %s: %g', key, history[key].loc[best])

def reload_last_checkpoint(checkpoint_format, n_epochs, distributed):
    # TODO: clarify confusion on inconsistent epoch numbering
    for epoch in range(n_epochs, 0, -1):
        checkpoint = checkpoint_format.format(epoch=epoch)
        if os.path.exists(checkpoint):
            # Fix for Lambda layer warning
            import models.cosmoflow
            # Use horovod's reload to prepare the DistributedOptimizer
            if distributed:
                model = hvd.load_model(checkpoint)
            else:
                model = tf.keras.models.load_model(checkpoint)
            return epoch, model
    raise Exception('Unable to find a checkpoint file at %s' % checkpoint_format)

def main():
    """Main function"""

    # Initialization
    args = parse_args()
    rank, local_rank, n_ranks = init_workers(args.distributed)
    config = load_config(args.config)
    output_dir = os.path.expandvars(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    config_logging(verbose=args.verbose)
    logging.info('Initialized rank %i local_rank %i size %i',
                 rank, local_rank, n_ranks)
    if rank == 0:
        logging.info('Configuration: %s', config)

    # Device and session configuration
    gpu = local_rank if args.rank_gpu else None
    configure_session(gpu=gpu, **config.get('device', {}))

    # Load the data
    data_config = config['data']
    if rank == 0:
        logging.info('Loading data')
    datasets = get_datasets(rank=rank, n_ranks=n_ranks, **data_config)
    logging.info('Datasets: %s', datasets)

    # Construct or reload the model
    if rank == 0:
        logging.info('Building the model')
    initial_epoch = 0
    checkpoint_format = os.path.join(output_dir, 'checkpoint-{epoch:03d}.h5')
    if args.resume:
        # Reload model from last checkpoint
        initial_epoch, model = reload_last_checkpoint(
            checkpoint_format, data_config['n_epochs'],
            distributed=args.distributed)
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
    if rank == 0:
        logging.info('Preparing callbacks')
    callbacks = []
    if args.distributed:

        # Broadcast initial variable states from rank 0 to all processes.
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

        # Average metrics across workers
        callbacks.append(hvd.callbacks.MetricAverageCallback())

        # Learning rate warmup
        train_config = config['train']
        warmup_epochs = train_config.get('lr_warmup_epochs', 0)
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(
            warmup_epochs=warmup_epochs, verbose=1))

        # Learning rate decay schedule
        for lr_schedule in train_config.get('lr_schedule', []):
            if rank == 0:
                logging.info('Adding LR schedule: %s', lr_schedule)
            callbacks.append(hvd.callbacks.LearningRateScheduleCallback(**lr_schedule))

    # Timing
    timing_callback = TimingCallback()
    callbacks.append(timing_callback)

    # Checkpointing and CSV logging from rank 0 only
    if rank == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_format))
        callbacks.append(tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, 'history.csv'), append=args.resume))

    if rank == 0:
        logging.debug('Callbacks: %s', callbacks)

    # Train the model
    if rank == 0:
        logging.info('Beginning training')
    fit_verbose = 1 if (args.verbose and rank==0) else 2
    model.fit(datasets['train_dataset'],
              steps_per_epoch=datasets['n_train_steps'],
              epochs=data_config['n_epochs'],
              validation_data=datasets['valid_dataset'],
              validation_steps=datasets['n_valid_steps'],
              callbacks=callbacks,
              initial_epoch=initial_epoch,
              verbose=fit_verbose)

    # Print training summary
    if rank == 0:
        print_training_summary(output_dir)

    # Finalize
    if rank == 0:
        logging.info('All done!')

if __name__ == '__main__':
    main()
