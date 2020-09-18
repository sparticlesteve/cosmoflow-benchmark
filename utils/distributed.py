"""Utilties for distributed processing"""

import horovod.tensorflow.keras as hvd

def rank():
    try:
        return hvd.rank()
    except ValueError:
        return 0

def barrier():
    try:
        hvd.allreduce([], name='Barrier')
    except ValueError:
        pass
