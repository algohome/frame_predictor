# Imports
import argparse
import numpy as np
from PIL import Image, ImageOps
import glob
import os
import sys
import pathlib
import keras
from tensorflow.keras.layers import BatchNormalization, Activation
from models import *
from FrameProducer import FrameProducer
from DataFeeder import DataFeeder, MP4Feeder
from utils import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Parser
parser = argparse.ArgumentParser(description="Processes a video")

parser.add_argument('input_video', type=str, help='path to input video')

parser.add_argument('output_path', type=str, help='path to save output mp4')

parser.add_argument('--height', type=int, help='height to cast to (in pixels)', default=512, required=False)
parser.add_argument('--width', type=int, help='width to cast to (in pixels)', default=512, required=False)

parser.add_argument('--mode', type=str, help='color space to use', default="RGB", choices=['RGB','YCbCr'], required=False)

parser.add_argument('--frames', type=int, help='number of frames to produce', default = None, required=False)
parser.add_argument('--fps', type=int, help='frames per second of output', default = 30, required=False)

parser.add_argument('--c', type=float, help='used in calculations', default=0.5, required=False)
parser.add_argument('--history', type=int, help='used in calculations', default=3, required=False)

parser.add_argument('--model', type=str, help='name of model to use', default="darkart", choices=['darkart','bone','road'], required=False)

parser.add_argument('--preset', type=int, help='overrides other settings with some presets', default=None, choices=[None, 1, 2, 3], required=False)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    if args.preset == 1:
        args.height = 2048
        args.width = 2048
        args.c = 0.5
        args.history = 0
        args.model = 'bone'

    if args.preset == 2:
        args.height = 2048
        args.width = 2048
        args.c = 0.25
        args.history = 2
        args.model = 'darkart'

    if args.preset == 3:
        args.height = 2048
        args.width = 2048
        args.c = 1.75
        args.history = 2
        args.model = 'road'


    if args.model == 'darkart':
        model_fn = make_darkart_model
        weights = 'model_weights/darkart'
    elif args.model == 'bone':
        model_fn = make_bone_model
        weights = 'model_weights/bone'
    elif args.model == 'road':
        model_fn = make_road_model
        weights = 'model_weights/road'


    df = MP4Feeder(args.input_video,
                    (args.width,args.height,3),
                    args.mode,
                    5,
                    normalization_function = lambda x : np.array([p/255 for p in [r for r in x]]),
                    preprocess_function = None)

    fp = FrameProducer(model_fn, args.c, weights, args.history, df, 5, transitions=[], frames_to_produce=args.frames, save_dir='outputs/frames', mode="RGB", output_fps = args.fps, vid_output_name=args.output_path)

    fp.go()
