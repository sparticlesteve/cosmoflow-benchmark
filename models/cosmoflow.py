"""Model specification for CosmoFlow"""

import tensorflow as tf
import tensorflow.keras.layers as layers

def build_model(input_shape, target_size, dropout=0):
    """Construct the CosmoFlow 3D CNN model"""

    conv_args = dict(kernel_size=3, padding='same', activation='relu')

    model = tf.keras.models.Sequential([

        layers.Conv3D(16, input_shape=input_shape, **conv_args),
        layers.MaxPool3D(pool_size=2),

        layers.Conv3D(16, **conv_args),
        layers.MaxPool3D(pool_size=2),

        layers.Conv3D(16, **conv_args),
        layers.MaxPool3D(pool_size=2),

        layers.Conv3D(16, **conv_args),
        layers.MaxPool3D(pool_size=2),

        layers.Conv3D(16, **conv_args),
        layers.MaxPool3D(pool_size=2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout),

        layers.Dense(64, activation='relu'),
        layers.Dropout(dropout),

        layers.Dense(target_size)
    ])

    return model
