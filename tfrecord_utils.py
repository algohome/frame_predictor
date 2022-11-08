import tensorflow as tf
import tensorflow_io as tfio
import keras
from tensorflow.keras import layers

from keras import Model
from keras.utils import plot_model
import PIL
from PIL import Image, ImageFilter
import numpy as np
import os
import pathlib
import glob


def _parse_function(proto):
    keys_to_features = {
        'image1':tf.io.FixedLenFeature([], tf.string),
        'image2':tf.io.FixedLenFeature([], tf.string),
        'image3':tf.io.FixedLenFeature([], tf.string),
        'image4':tf.io.FixedLenFeature([], tf.string),
        'image5':tf.io.FixedLenFeature([], tf.string),
        'y':tf.io.FixedLenFeature([], tf.string)
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    parsed_features['image1'] = tf.io.decode_raw(parsed_features['image'], tf.float64)
    parsed_features['label'] = tf.io.decode_raw(parsed_features['label'], tf.float64)

    print(parsed_features['image'])

    parsed_features['image'] = tf.reshape(parsed_features['image'], [512,512,15])
    parsed_features['label'] = tf.reshape(parsed_features['label'], [512,512,3])

    print(parsed_features['image'])

    return parsed_features['image'], parsed_features['label']

# READING TFRECORD
def parse_tfrecord_fn(example):
    feature_description = {
        "image1": tf.io.FixedLenFeature([], tf.string),
        "image2": tf.io.FixedLenFeature([], tf.string),
        "image3": tf.io.FixedLenFeature([], tf.string),
        "image4": tf.io.FixedLenFeature([], tf.string),
        "image5": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image1"] = tf.io.decode_jpeg(example["image1"], channels=3)
    example["image2"] = tf.io.decode_jpeg(example["image2"], channels=3)
    example["image3"] = tf.io.decode_jpeg(example["image3"], channels=3)
    example["image4"] = tf.io.decode_jpeg(example["image4"], channels=3)
    example["image5"] = tf.io.decode_jpeg(example["image5"], channels=3)
    
    formatted_example = {}

    x = tf.concat([example["image1"],
                 example["image2"],
                 example["image3"],
                 example["image4"],
                 example["image5"]], axis = -1) / 255
    
    
    formatted_example['image'] = x
    
    formatted_example["label"] = tf.io.decode_jpeg(example["y"], channels=3) / 255

    return (formatted_example['image'], formatted_example['label'])

def parse_tfrecord_fn_yuv(example):
    feature_description = {
        "image1": tf.io.FixedLenFeature([], tf.string),
        "image2": tf.io.FixedLenFeature([], tf.string),
        "image3": tf.io.FixedLenFeature([], tf.string),
        "image4": tf.io.FixedLenFeature([], tf.string),
        "image5": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image1"] = tfio.experimental.color.rgb_to_ycbcr(tf.io.decode_jpeg(example["image1"], channels=3))
    example["image2"] = tfio.experimental.color.rgb_to_ycbcr(tf.io.decode_jpeg(example["image2"], channels=3))
    example["image3"] = tfio.experimental.color.rgb_to_ycbcr(tf.io.decode_jpeg(example["image3"], channels=3))
    example["image4"] = tfio.experimental.color.rgb_to_ycbcr(tf.io.decode_jpeg(example["image4"], channels=3))
    example["image5"] = tfio.experimental.color.rgb_to_ycbcr(tf.io.decode_jpeg(example["image5"], channels=3))
    
    formatted_example = {}
    #print(example)
    #print(example["image1"].numpy())
    print(example)
    x = tf.concat([example["image1"],
                 example["image2"],
                 example["image3"],
                 example["image4"],
                 example["image5"]], axis = -1) / 255
    
    
    formatted_example['image'] = x
    
    formatted_example["label"] = tfio.experimental.color.rgb_to_ycbcr(tf.io.decode_jpeg(example["y"], channels=3)) / 255
    #print(formatted_example)
    #return formatted_example
    return (formatted_example['image'], formatted_example['label'])


def load_dataset(path, yuv=False):
    dataset = tf.data.TFRecordDataset(path)
    
    if not yuv:
        dataset = dataset.map(parse_tfrecord_fn)
    if yuv:
        dataset = dataset.map(parse_tfrecord_fn_yuv)

    return dataset

def get_dataset(path, shuffle=160, batch=4, yuv=False):
    dataset = load_dataset(path, yuv)

    dataset = dataset.shuffle(shuffle)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch)

    return dataset