import tensorflow as tf
from tensorflow.keras import layers
import keras
from keras import Model
import PIL
from PIL import Image, ImageFilter
import numpy as np
import os
import pathlib
import glob

from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras import regularizers


def sin_act(x):
    return K.sin(x)


def cos_act(x):
    return K.cos(x)


def make_small_model_1(input_shape=(512, 512, 15)):
    inputs = layers.Input(input_shape, name="image")
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    encoded = x

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        32, (3, 3), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        32, (3, 3), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    decoded = layers.Conv2D(
        3, (3, 3), padding="same", activation="sigmoid", dtype="float32", name="label"
    )(x)

    outs = decoded

    ae = keras.Model(inputs, outs)
    return ae


def make_model_1(input_shape=(512, 512, 15)):
    inputs = layers.Input(input_shape, name="image")
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    encoded = x

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (3, 3), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (3, 3), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        16, (3, 3), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    decoded = layers.Conv2D(
        3, (3, 3), padding="same", activation="sigmoid", dtype="float32", name="label"
    )(x)

    outs = decoded

    ae = keras.Model(inputs, outs)
    return ae


def make_large_model_1(input_shape=(512, 512, 15)):
    inputs = layers.Input(input_shape, name="image")
    x = layers.Conv2D(16, (7, 7), padding="same", activation="relu")(inputs)
    x = layers.Conv2D(16, (5, 5), padding="same", activation="relu")(x)
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    encoded = x

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (3, 3), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (3, 3), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (3, 3), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (3, 3), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        16, (3, 3), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(16, (5, 5), padding="same", activation="relu")(x)
    x = layers.Conv2D(16, (7, 7), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    decoded = layers.Conv2D(
        3, (3, 3), padding="same", activation="sigmoid", dtype="float32", name="label"
    )(x)

    outs = decoded

    ae = keras.Model(inputs, outs)
    return ae


def make_large_model_2(input_shape=(512, 512, 15)):
    inputs = layers.Input(input_shape, name="image")
    x = layers.Conv2D(32, (7, 7), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(256, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    encoded = x

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (3, 3), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (3, 3), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (3, 3), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (3, 3), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        16, (3, 3), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(16, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(16, (7, 7), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    decoded = layers.Conv2D(
        3, (3, 3), padding="same", activation="sigmoid", dtype="float32", name="label"
    )(x)

    outs = decoded

    ae = keras.Model(inputs, outs)
    return ae


def make_deep_model_1(input_shape=(512, 512, 15)):
    inputs = layers.Input(input_shape, name="image")
    x = layers.Conv2D(8, (5, 5), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 256

    x = layers.Conv2D(16, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 128

    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 64

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)  # 32

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 16

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 8

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    preflat = x

    x = layers.Flatten()(x)  # FLAT

    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    # x = layers.Dense(64, activation='relu')(x)
    # x = layers.BatchNormalization(axis=-1)(x)

    encoded = x

    x = layers.Dense(128, activation="relu")(encoded)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(2048, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Reshape((8, 8, 32))(x)  # 8

    x = layers.Concatenate(axis=-1)([x, preflat])

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        256, (3, 3), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 16
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        128, (3, 3), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 32
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (5, 5), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 64
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(128, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        128, (3, 3), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 128
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (3, 3), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 256
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        32, (3, 3), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 512
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (7, 7), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(16, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(8, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    decoded = layers.Conv2D(
        3, (3, 3), padding="same", activation="sigmoid", dtype="float32", name="label"
    )(x)

    outs = decoded

    ae = keras.Model(inputs, outs)
    return ae


def make_deep_model_2(input_shape=(512, 512, 15)):
    inputs = layers.Input(input_shape, name="image")
    x = layers.Conv2D(8, (5, 5), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 256

    x = layers.Conv2D(16, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 128

    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 64

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)  # 32

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    s_32 = x

    x = layers.Conv2D(128, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 16

    s_16 = x

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 8

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    preflat = x

    x = layers.Flatten()(x)  # FLAT

    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    # x = layers.Dense(64, activation='relu')(x)
    # x = layers.BatchNormalization(axis=-1)(x)

    encoded = x

    x = layers.Dense(128, activation="relu")(encoded)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(2048, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Reshape((8, 8, 32))(x)  # 8

    x = layers.Concatenate(axis=-1)([x, preflat])

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        256, (3, 3), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 16
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Concatenate(axis=-1)([x, s_16])

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        128, (3, 3), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 32
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Concatenate(axis=-1)([x, s_32])

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (5, 5), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 64
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(128, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        128, (3, 3), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 128
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (3, 3), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 256
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        32, (3, 3), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 512
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (7, 7), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(16, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(8, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    decoded = layers.Conv2D(
        3, (3, 3), padding="same", activation="sigmoid", dtype="float32", name="label"
    )(x)

    outs = decoded

    ae = keras.Model(inputs, outs)
    return ae


def make_deep_model_3(input_shape=(512, 512, 15)):
    inputs = layers.Input(input_shape, name="image")
    x = layers.Conv2D(8, (5, 5), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 256

    x = layers.Conv2D(16, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 128

    s_128 = x

    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 64

    s_64 = x

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)  # 32

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    s_32 = x

    x = layers.Conv2D(128, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 16

    s_16 = x

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 8

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    preflat = x

    x = layers.Flatten()(x)  # FLAT

    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    # x = layers.Dense(64, activation='relu')(x)
    # x = layers.BatchNormalization(axis=-1)(x)

    encoded = x

    x = layers.Dense(128, activation="relu")(encoded)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(2048, activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Reshape((8, 8, 32))(x)  # 8

    x = layers.Concatenate(axis=-1)([x, preflat])

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        256, (3, 3), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 16
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Concatenate(axis=-1)([x, s_16])

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        128, (3, 3), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 32
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Concatenate(axis=-1)([x, s_32])

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (5, 5), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 64
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Concatenate(axis=-1)([x, s_64])

    x = layers.Conv2D(128, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        128, (3, 3), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 128
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Concatenate(axis=-1)([x, s_128])

    x = layers.Conv2D(64, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        64, (3, 3), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 256
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(
        32, (3, 3), padding="same", strides=2, activation="relu"
    )(
        x
    )  # 512
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (7, 7), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(16, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(8, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    decoded = layers.Conv2D(
        3, (3, 3), padding="same", activation="sigmoid", dtype="float32", name="label"
    )(x)

    outs = decoded

    ae = keras.Model(inputs, outs)
    return ae


def make_deep_model_4(input_shape=(512, 512, 15)):
    inputs = layers.Input(input_shape, name="image")
    x = layers.Conv2D(8, (5, 5), padding="same", activation=None)(inputs)

    x = tf.keras.layers.PReLU()(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 256

    x = layers.Conv2D(16, (5, 5), padding="same", activation=None)(x)

    x = tf.keras.layers.PReLU()(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 128

    s_128 = x

    x = layers.Conv2D(32, (5, 5), padding="same", activation=None)(x)

    x = tf.keras.layers.PReLU()(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 64

    s_64 = x

    x = layers.Conv2D(64, (5, 5), padding="same", activation=None)(x)

    x = tf.keras.layers.PReLU()(x)
    x = layers.BatchNormalization(axis=-1)(x)  # 32

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    s_32 = x

    x = layers.Conv2D(128, (5, 5), padding="same", activation=None)(x)

    x = tf.keras.layers.PReLU()(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 16

    s_16 = x

    x = layers.Conv2D(256, (3, 3), padding="same", activation=None)(x)

    x = tf.keras.layers.PReLU()(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2))(x)  # 8

    x = layers.Conv2D(256, (3, 3), padding="same", activation=None)(x)

    x = tf.keras.layers.PReLU()(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation=None)(x)

    x = tf.keras.layers.PReLU()(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation=None)(x)

    x = tf.keras.layers.PReLU()(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation=None)(x)

    x = tf.keras.layers.PReLU()(x)
    x = layers.BatchNormalization(axis=-1)(x)

    preflat = x

    x = layers.Flatten()(x)  # FLAT

    x = layers.Dense(512, activation="tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(256, activation="tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(128, activation="tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(64, activation="tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    encoded = x

    x = layers.Dense(128, activation="tanh")(encoded)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(256, activation="tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(512, activation="tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Dense(2048, activation="tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Reshape((8, 8, 32))(x)  # 8

    x = layers.Concatenate(axis=-1)([x, preflat])

    x = layers.Conv2D(256, (3, 3), padding="same", activation=None)(x)
    x = tf.keras.layers.PReLU()(x)

    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(256, (3, 3), padding="same", strides=2, activation=None)(
        x
    )
    x = tf.keras.layers.PReLU()(x)

    # 16
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Concatenate(axis=-1)([x, s_16])

    x = layers.Conv2D(128, (3, 3), padding="same", activation=None)(x)
    x = tf.keras.layers.PReLU()(x)

    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(128, (3, 3), padding="same", strides=2, activation=None)(
        x
    )
    x = tf.keras.layers.PReLU()(x)

    # 32
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Concatenate(axis=-1)([x, s_32])

    x = layers.Conv2D(64, (5, 5), padding="same", activation=None)(x)
    x = tf.keras.layers.PReLU()(x)

    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(64, (5, 5), padding="same", strides=2, activation=None)(
        x
    )
    x = tf.keras.layers.PReLU()(x)

    # 64
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Concatenate(axis=-1)([x, s_64])

    x = layers.Conv2D(128, (5, 5), padding="same", activation=None)(x)
    x = tf.keras.layers.PReLU()(x)

    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(128, (3, 3), padding="same", strides=2, activation=None)(
        x
    )
    x = tf.keras.layers.PReLU()(x)

    # 128
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Concatenate(axis=-1)([x, s_128])

    x = layers.Conv2D(64, (5, 5), padding="same", activation=None)(x)
    x = tf.keras.layers.PReLU()(x)

    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(64, (3, 3), padding="same", strides=2, activation=None)(
        x
    )
    x = tf.keras.layers.PReLU()(x)

    # 256
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (5, 5), padding="same", activation=None)(x)
    x = tf.keras.layers.PReLU()(x)

    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2DTranspose(32, (3, 3), padding="same", strides=2, activation=None)(
        x
    )
    x = tf.keras.layers.PReLU()(x)

    # 512
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (7, 7), padding="same", activation=None)(x)
    x = tf.keras.layers.PReLU()(x)

    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(16, (5, 5), padding="same", activation=None)(x)
    x = tf.keras.layers.PReLU()(x)

    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(8, (3, 3), padding="same", activation=None)(x)
    x = tf.keras.layers.PReLU()(x)

    x = layers.BatchNormalization(axis=-1)(x)

    decoded = layers.Conv2D(
        3, (3, 3), padding="same", activation="sigmoid", dtype="float32", name="label"
    )(x)

    outs = decoded

    ae = keras.Model(inputs, outs)
    return ae


def make_large_split_1(input_shape=(512, 512, 15)):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(64, (3, 3), padding="same")(inputs)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (1, 1), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # 128
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (1, 1), padding="same")(x)
    x = Activation("relu")(x)
    y_encoded = layers.BatchNormalization(axis=-1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(y_encoded)

    # 64
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (1, 1), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (1, 1), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # 32
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (1, 1), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    # 16
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (1, 1), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding="same")(x)

    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (1, 1), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    # 8
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(8, (1, 1), padding="same")(x)
    color_encoded = Activation("relu")(x)

    x = layers.Conv2D(128, (3, 3), padding="same")(color_encoded)
    x = Activation("tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(64, (5, 5), padding="same", strides=2)(x)
    x = Activation("tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = Activation("tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(32, (5, 5), padding="same", strides=2)(x)
    x = Activation("tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(64, (5, 5), padding="same", strides=2)(x)
    x = Activation("tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(64, (5, 5), padding="same", strides=2)(x)
    x = Activation("tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(64, (5, 5), padding="same", strides=2)(x)
    x = Activation("tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(64, (5, 5), padding="same", strides=2)(x)
    x = Activation("tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = Activation("tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same")(x)
    x = Activation("tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(16, (7, 7), padding="same")(x)
    x = Activation("tanh")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    color_decoded = layers.Conv2D(
        2, (7, 7), padding="same", activation="sigmoid", dtype="float32"
    )(x)

    x = layers.Conv2D(128, (5, 5), padding="same")(y_encoded)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(32, (5, 5), padding="same", strides=2)(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (5, 5), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(16, (5, 5), padding="same")(x)
    x = Activation("relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    y_decoded = layers.Conv2D(
        1, (7, 7), padding="same", activation="sigmoid", dtype="float32"
    )(x)

    decoded = layers.Concatenate(axis=-1)([y_decoded, color_decoded])

    outs = decoded

    ae = keras.Model(inputs, outs)
    return ae


def make_darkart_model(input_shape=(512, 512, 15)):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(32, (7, 7), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    # skip_one = x
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (7, 7), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    skip_two = x
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (7, 7), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    # x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    # x = layers.MaxPooling2D(pool_size=(2,2))(x)

    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    # x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    # x = layers.MaxPooling2D(pool_size=(2,2))(x)

    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    encoded = layers.Conv2D(16, (1, 1), padding="same", activation="relu")(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(encoded)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(
        32, (5, 5), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    # x = layers.Add()([x, once_encoded])

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(
        32, (7, 7), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Concatenate(axis=-1)([x, skip_two])
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(
        32, (7, 7), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    # x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    # x = layers.Conv2DTranspose(32, (3,3), padding='same', strides=2, activation='relu')(x)
    # x = layers.BatchNormalization(axis=-1)(x)

    # x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    # x = layers.Conv2DTranspose(32, (3,3), padding='same', strides=2, activation='relu')(x)
    # x = layers.BatchNormalization(axis=-1)(x)

    # x = layers.Concatenate(axis=-1)([x, skip_one])
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (7, 7), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    decoded = layers.Conv2D(
        3, (7, 7), padding="same", activation="sigmoid", dtype="float32"
    )(x)

    outs = decoded

    ae = keras.Model(inputs, outs)
    return ae


def make_road_model(input_shape=(512, 512, 15)):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(64, (7, 7), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    skip_one = x
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (7, 7), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    skip_two = x
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    # x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    # x = layers.MaxPooling2D(pool_size=(2,2))(x)

    # x = layers.BatchNormalization(axis=-1)(x)
    # x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    # x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    # x = layers.MaxPooling2D(pool_size=(2,2))(x)

    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    encoded = layers.Conv2D(16, (1, 1), padding="same", activation="relu")(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(encoded)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(
        32, (5, 5), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    # x = layers.Add()([x, once_encoded])

    x = layers.Concatenate(axis=-1)([x, skip_two])
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(
        32, (7, 7), padding="same", strides=2, activation="relu"
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    # x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    # x = layers.Conv2DTranspose(32, (3,3), padding='same', strides=2, activation='relu')(x)
    # x = layers.BatchNormalization(axis=-1)(x)

    # x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    # x = layers.Conv2DTranspose(32, (3,3), padding='same', strides=2, activation='relu')(x)
    # x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Concatenate(axis=-1)([x, skip_one])
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization(axis=-1)(x)
    decoded = layers.Conv2D(
        3, (7, 7), padding="same", activation="sigmoid", dtype="float32"
    )(x)

    outs = decoded

    ae = keras.Model(inputs, outs)
    return ae


def make_bone_model(input_shape=(512, 512, 15)):
    inputs = layers.Input(input_shape)
    x = layers.Conv2D(32, (7, 7), padding="same", activation=sin_act)(inputs)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (7, 7), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    skip_two = x
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (7, 7), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (7, 7), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(32, (5, 5), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    encoded = layers.Conv2D(16, (1, 1), padding="same", activation=sin_act)(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation=sin_act)(encoded)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(
        32, (5, 5), padding="same", strides=2, activation=sin_act
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (5, 5), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(
        32, (7, 7), padding="same", strides=2, activation=sin_act
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (5, 5), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(
        32, (7, 7), padding="same", strides=2, activation=sin_act
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Concatenate(axis=-1)([x, skip_two])
    x = layers.Conv2D(32, (3, 3), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2DTranspose(
        32, (7, 7), padding="same", strides=2, activation=sin_act
    )(x)
    x = layers.BatchNormalization(axis=-1)(x)

    x = layers.Conv2D(32, (3, 3), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (5, 5), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.Conv2D(32, (7, 7), padding="same", activation=sin_act)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    decoded = layers.Conv2D(
        3, (7, 7), padding="same", activation="sigmoid", dtype="float32"
    )(x)

    outs = decoded

    ae = keras.Model(inputs, outs)
    return ae
