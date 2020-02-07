"""Model specification for CosmoFlow"""

import tensorflow as tf
import tensorflow.keras.layers as layers

def scale_1p2(x):
    return x*1.2

def build_model(input_shape, target_size,
                conv_sizes=[16, 16, 16, 16, 16],
                kernel_size=2,
                fc_sizes=[128, 64],
                hidden_activation='LeakyReLU',
                pooling_type='MaxPool3D',
                dropout=0):
    """Construct the CosmoFlow 3D CNN model"""

    conv_args = dict(kernel_size=kernel_size, padding='valid')
    hidden_activation = getattr(layers, hidden_activation)
    pooling_type = getattr(layers, pooling_type)

    model = tf.keras.models.Sequential()

    # First convolutional layer
    model.add(layers.Conv3D(conv_sizes[0], input_shape=input_shape, **conv_args))
    model.add(hidden_activation())
    model.add(pooling_type(pool_size=2))

    # Additional conv layers
    for conv_size in conv_sizes[1:]:
        model.add(layers.Conv3D(conv_size, **conv_args))
        model.add(hidden_activation())
        model.add(pooling_type(pool_size=2))

    model.add(layers.Flatten())

    # Fully-connected layers
    for fc_size in fc_sizes:
        model.add(layers.Dense(fc_size))
        model.add(hidden_activation())
        model.add(layers.Dropout(dropout))

    # Output layers
    model.add(layers.Dense(target_size, activation='tanh'))
    model.add(layers.Lambda(scale_1p2))

    return model
