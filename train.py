"""
Main training script for the CosmoFlow Keras benchmark
"""

# System imports
import os
import argparse
import logging
import pickle
from types import SimpleNamespace

# External imports
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(logging.ERROR)
import horovod.tensorflow.keras as hvd

# Local imports
from data import get_datasets
from models import get_model
# Fix for loading Lambda layer checkpoints
from models.layers import *
from utils.optimizers import get_optimizer
from utils.callbacks import TimingCallback
from utils.device import configure_session
from utils.argparse import ReadYaml

# Stupid workaround until absl logging fix, see:
# https://github.com/tensorflow/tensorflow/issues/26691
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/cosmo.yaml')
    add_arg('--output-dir', help='Override output directory')

    # Override data settings
    add_arg('--data-dir', help='Override the path to input files')
    add_arg('--n-train', type=int, help='Override number of training samples')
    add_arg('--n-valid', type=int, help='Override number of validation samples')
    add_arg('--batch-size', type=int, help='Override the batch size')
    add_arg('--n-epochs', type=int, help='Override number of epochs')
    add_arg('--apply-log', type=int, choices=[0, 1], help='Apply log transform to data')
    add_arg('--shard-type', choices=['global', 'local'],
            help='Override file sharding config')

    # Hyperparameter settings
    add_arg('--conv-size', type=int, help='CNN size parameter')
    add_arg('--fc1-size', type=int, help='Fully-connected size parameter 1')
    add_arg('--fc2-size', type=int, help='Fully-connected size parameter 2')
    add_arg('--hidden-activation', help='Override hidden activation function')
    add_arg('--dropout', type=float, help='Override dropout')
    add_arg('--optimizer', help='Override optimizer type')
    add_arg('--lr', type=float, help='Override learning rate')

    # Other settings
    add_arg('-d', '--distributed', action='store_true')
    add_arg('--rank-gpu', action='store_true',
            help='Use GPU based on local rank')
    add_arg('--resume', action='store_true',
            help='Resume from last checkpoint')
    add_arg('--print-fom', action='store_true',
            help='Print parsable figure of merit')
    add_arg('-v', '--verbose', action='store_true')
    return parser.parse_args()

def init_workers(distributed=False):
    if distributed:
        hvd.init()
        return SimpleNamespace(rank=hvd.rank(), size=hvd.size(),
                               local_rank=hvd.local_rank(),
                               local_size=hvd.local_size())
    else:
        return SimpleNamespace(rank=0, size=1, local_rank=0, local_size=1)

def config_logging(verbose):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)

def load_config(args):
    """Reads the YAML config file and returns a config dictionary"""
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Expand paths
    output_dir = config['output_dir'] if args.output_dir is None else args.output_dir
    config['output_dir'] = os.path.expandvars(output_dir)

    # Override data config from command line
    if args.data_dir is not None:
        config['data']['data_dir'] = args.data_dir
    if args.n_train is not None:
        config['data']['n_train'] = args.n_train
    if args.n_valid is not None:
        config['data']['n_valid'] = args.n_valid
    if args.batch_size is not None:
        config['data']['batch_size'] = args.batch_size
    if args.n_epochs is not None:
        config['data']['n_epochs'] = args.n_epochs
    if args.apply_log is not None:
        config['data']['apply_log'] = bool(args.apply_log)
    if args.shard_type is not None:
        config['data']['shard_type'] = args.shard_type

    # Hyperparameters
    if args.conv_size is not None:
        config['model']['conv_size'] = args.conv_size
    if args.fc1_size is not None:
        config['model']['fc1_size'] = args.fc1_size
    if args.fc2_size is not None:
        config['model']['fc2_size'] = args.fc2_size
    if args.hidden_activation is not None:
        config['model']['hidden_activation'] = args.hidden_activation
    if args.dropout is not None:
        config['model']['dropout'] = args.dropout
    if args.optimizer is not None:
        config['optimizer']['name'] = args.optimizer
    if args.lr is not None:
        config['optimizer']['lr'] = args.lr

    return config

def save_config(config):
    output_dir = config['output_dir']
    config_file = os.path.join(output_dir, 'config.pkl')
    logging.info('Writing config via pickle to %s', config_file)
    with open(config_file, 'wb') as f:
        pickle.dump(config, f)

def load_history(output_dir):
    return pd.read_csv(os.path.join(output_dir, 'history.csv'))

def print_training_summary(output_dir, print_fom):
    history = load_history(output_dir)
    if 'val_loss' in history.keys():
        best = history.val_loss.idxmin()
        logging.info('Best result:')
        for key in history.keys():
            logging.info('  %s: %g', key, history[key].loc[best])
        # Figure of merit printing for HPO parsing
        if print_fom:
            print('FoM:', history['val_loss'].loc[best])

def reload_last_checkpoint(checkpoint_format, n_epochs, distributed):
    """Finds and loads the last checkpoint matching the provided pattern"""
    # Count down from n_epochs to 0 to find the last epoch.
    # Note that keras names checkpoint files with epoch number starting from 1.
    # So the matched number corresponds to the new initial epoch.
    for epoch in range(n_epochs, 0, -1):
        checkpoint = checkpoint_format.format(epoch=epoch)
        if os.path.exists(checkpoint):
            logging.info('Found last checkpoint at %s', checkpoint)
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
    dist = init_workers(args.distributed)
    config = load_config(args)
    os.makedirs(config['output_dir'], exist_ok=True)
    config_logging(verbose=args.verbose)
    logging.info('Initialized rank %i size %i local_rank %i local_size %i',
                 dist.rank, dist.size, dist.local_rank, dist.local_size)
    if dist.rank == 0:
        logging.info('Configuration: %s', config)

    # Device and session configuration
    gpu = dist.local_rank if args.rank_gpu else None
    if gpu is not None:
        logging.info('Taking gpu %i', gpu)
    configure_session(gpu=gpu, **config.get('device', {}))

    # Load the data
    data_config = config['data']
    if dist.rank == 0:
        logging.info('Loading data')
    datasets = get_datasets(dist=dist, **data_config)
    logging.debug('Datasets: %s', datasets)

    # Construct or reload the model
    if dist.rank == 0:
        logging.info('Building the model')
    initial_epoch = 0
    checkpoint_format = os.path.join(config['output_dir'], 'checkpoint-{epoch:03d}.h5')
    if args.resume and os.path.exists(checkpoint_format.format(epoch=1)):
        # Reload model from last checkpoint
        initial_epoch, model = reload_last_checkpoint(
            checkpoint_format, data_config['n_epochs'],
            distributed=args.distributed)
    else:
        # Build a new model
        model = get_model(**config['model'])
        # Configure the optimizer
        opt = get_optimizer(n_ranks=dist.size,
                            distributed=args.distributed,
                            **config['optimizer'])
        # Compile the model
        train_config = config['train']
        model.compile(optimizer=opt, loss=train_config['loss'],
                      metrics=train_config['metrics'])

    if dist.rank == 0:
        model.summary()

    # Save configuration to output directory
    if dist.rank == 0:
        config['n_ranks'] = dist.size
        data_config['n_train'] = datasets['n_train']
        data_config['n_valid'] = datasets['n_valid']
        save_config(config)

    # Prepare the callbacks
    if dist.rank == 0:
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
    lr_schedule = train_config.get('lr_schedule', {})
    if dist.rank == 0:
        logging.info('Adding LR decay schedule: %s', lr_schedule)
    callbacks.append(tf.keras.callbacks.LearningRateScheduler(
        schedule=lambda epoch, lr: lr * lr_schedule.get(epoch, 1)))

    # Timing
    timing_callback = TimingCallback()
    callbacks.append(timing_callback)

    # Checkpointing and logging from rank 0 only
    if dist.rank == 0:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_format))
        callbacks.append(tf.keras.callbacks.CSVLogger(
            os.path.join(config['output_dir'], 'history.csv'), append=args.resume))
        callbacks.append(tf.keras.callbacks.TensorBoard(
            os.path.join(config['output_dir'], 'tensorboard')))

    # Early stopping
    patience = config.get('early_stopping_patience', None)
    if patience is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', min_delta=1e-5, patience=patience, verbose=1))

    if dist.rank == 0:
        logging.debug('Callbacks: %s', callbacks)

    # Train the model
    if dist.rank == 0:
        logging.info('Beginning training')
    fit_verbose = 1 if (args.verbose and dist.rank==0) else 2
    model.fit(datasets['train_dataset'],
              steps_per_epoch=datasets['n_train_steps'],
              epochs=data_config['n_epochs'],
              validation_data=datasets['valid_dataset'],
              validation_steps=datasets['n_valid_steps'],
              callbacks=callbacks,
              initial_epoch=initial_epoch,
              verbose=fit_verbose)

    # Print training summary
    if dist.rank == 0:
        print_training_summary(config['output_dir'], args.print_fom)

    # Finalize
    if dist.rank == 0:
        logging.info('All done!')

if __name__ == '__main__':
    main()
