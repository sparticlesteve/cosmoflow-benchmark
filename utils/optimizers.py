"""
Utilty code for constructing optimizers and scheduling learning rates.
"""

# System
import math
from functools import partial

# Externals
from tensorflow import keras
import horovod.tensorflow.keras as hvd
from mlperf_logging import mllog

# Locals
import utils.distributed

def _lr_schedule(epoch, base_lr, peak_lr, n_warmup_epochs, decay_schedule={}):
    """Learning rate schedule function.

    Gives the learning rate as a function of epoch according to
    additional settings:
        base_lr: baseline unscaled learning rate at beginning of training.
        peak_lr: scaled learning at end of warmup period
        n_warmup_epochs: number of linear warmup epochs
        decay_schedule: a dict of epoch number -> decay factor
    """
    # Linear LR warmup
    if epoch < n_warmup_epochs:
        return epoch * (peak_lr - base_lr) / n_warmup_epochs + base_lr
    else:
        # Find the most recent decay factor
        decay_factor = 1.
        decay_epoch = 0
        for e, d in decay_schedule.items():
            if e >= decay_epoch and e < epoch:
                decay_epoch, decay_factor = e, d
        return peak_lr * decay_factor

def get_lr_schedule(base_lr, global_batch_size, base_batch_size=None,
                    scaling=None, n_warmup_epochs=0, decay_schedule={}):
    """Get the learning rate schedule function"""
    if scaling == 'linear':
        scale_factor = global_batch_size / base_batch_size
    elif scaling == 'sqrt':
        scale_factor = math.sqrt(global_batch_size / base_batch_size)
    else:
        scale_factor = 1.;
    peak_lr = base_lr * scale_factor

    # MLPerf logging
    # NOTE: there is currently a confusing mismatch between the parameter
    # naming convention in this implementation and MLPerf's hyperparameter
    # conventions. Here we define base LR to be the LR at a baseline batch
    # size and the "peak" LR to be the value scaled according to current batch
    # size. We will leave things as-is for now.
    if utils.distributed.rank() == 0:
        mllogger = mllog.get_mllogger()
        mllogger.event(key=mllog.constants.OPT_BASE_LR, value=peak_lr)
        mllogger.event(key=mllog.constants.OPT_LR_WARMUP_EPOCHS, value=n_warmup_epochs)
        mllogger.event(key=mllog.constants.OPT_LR_WARMUP_FACTOR, value=scale_factor)
        mllogger.event(key=mllog.constants.OPT_LR_DECAY_BOUNDARY_EPOCHS,
                       value=sorted(decay_schedule.keys()))
        mllogger.event(key=mllog.constants.OPT_LR_DECAY_FACTOR,
                       value=max(decay_schedule.values()) if len(decay_schedule)>0 else 1)
    return partial(_lr_schedule, base_lr=base_lr, peak_lr=peak_lr,
                   n_warmup_epochs=n_warmup_epochs,
                   decay_schedule=decay_schedule)

def get_optimizer(name, distributed=False, **opt_args):
    """Configure the optimizer"""

    # MLPerf logging
    if utils.distributed.rank() == 0:
        mllogger = mllog.get_mllogger()
        mllogger.event(key=mllog.constants.OPT_NAME, value=name)

    # Construct the optimizer
    OptType = getattr(keras.optimizers, name)
    opt = OptType(**opt_args)

    # Distributed optimizer wrapper
    if distributed:
        opt = hvd.DistributedOptimizer(opt)

    return opt
