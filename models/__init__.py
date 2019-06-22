"""
Keras example model factory functions.
"""

def get_model(name, **model_args):
    if name == 'cosmoflow':
        from .cosmoflow import build_model
        return build_model(**model_args)
    else:
        raise ValueError('Model %s unknown' % name)
