# Imports
import numpy as np
from PIL import Image, ImageOps
import glob
import os
import sys
import pathlib
import keras
from tensorflow.keras.layers import BatchNormalization
from models import *
import cv2


class DataFeeder:
    def __init__(
        self,
        directory,
        img_dims,
        mode,
        n_frames,
        normalization_function=None,
        preprocess_function=None,
    ):
        assert mode in ["RGB", "YCbCr"]
        assert img_dims[-1] == 3

        self.start_frame = 0
        self.directory = directory
        self.img_dims = img_dims
        self.mode = mode
        self.n_frames = n_frames

        if normalization_function is not None:
            self.normalization_function = normalization_function
        else:
            self.normalization_function = lambda x: np.array(
                [p / 255 for p in [r for r in x]]
            )  # Works for arrays lol

        if preprocess_function is not None:
            self.preprocess_function = preprocess_function
        else:
            self.preprocess_function = lambda x: x

        self.files = glob.glob(os.path.join(directory, "*"))
        self.files.sort()
        print("Input frames:", len(self.files))

    # Do I want this function to return n images or an array?
    def get_n_frames(self, start_frame):
        output_array = np.zeros(
            [self.img_dims[0], self.img_dims[1], self.img_dims[2] * self.n_frames]
        )

        for i in range(self.n_frames):
            with Image.open(self.files[i + start_frame + self.start_frame]).convert(
                self.mode
            ).resize(self.img_dims[:-1]) as cur_frame:
                # cur_frame.show()
                cur_frame = self.preprocess(cur_frame)
                cur_frame = np.array(cur_frame)
                cur_frame = self.normalize(cur_frame)
                output_array[:, :, i * 3 : (i + 1) * (3)] = cur_frame

        return np.expand_dims(output_array, 0)

    def normalize(self, frame):
        return self.normalization_function(frame)

    def preprocess(self, frame):
        return self.preprocess_function(frame)


class MP4Feeder:
    def __init__(
        self,
        input_video,
        img_dims,
        mode,
        n_frames,
        normalization_function=None,
        preprocess_function=None,
    ):
        assert mode in ["RGB", "YCbCr"]
        assert img_dims[-1] == 3

        self.start_frame = 0
        self.input_video = input_video
        self.img_dims = img_dims
        self.mode = mode
        self.n_frames = n_frames

        if normalization_function is not None:
            self.normalization_function = normalization_function
        else:
            self.normalization_function = lambda x: np.array(
                [p / 255 for p in [r for r in x]]
            )  # Works for arrays lol

        if preprocess_function is not None:
            self.preprocess_function = preprocess_function
        else:
            self.preprocess_function = lambda x: x

        self.buffer = np.ones(
            (img_dims[1], img_dims[0], self.img_dims[2] * self.n_frames)
        )

        self.cap = cv2.VideoCapture(input_video)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n_input_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        for i in range(5):
            ret, frame = self.cap.read()
            if not ret:
                print("Something went wrong reading the video :(")
            else:
                frame = Image.fromarray(frame.astype("uint8")).resize(
                    (self.img_dims[0], self.img_dims[1])
                )
                frame = np.array(frame)
                frame = self.preprocess(frame)
                frame = self.normalize(frame)
                self.buffer[:, :, i * 3 : (i + 1) * (3)] = frame

    # Do I want this function to return n images or an array?
    def get_n_frames(self, start_frame):
        output_array = np.zeros(
            [self.img_dims[1], self.img_dims[0], self.img_dims[2] * self.n_frames]
        )
        output_array[:, :, : (self.n_frames - 1) * self.img_dims[2]] = self.buffer[
            :, :, self.img_dims[2] :
        ]

        ret, frame = self.cap.read()

        frame = Image.fromarray(frame.astype("uint8")).resize(self.img_dims[:-1])
        frame = np.array(frame)
        frame = self.preprocess(frame)
        frame = self.normalize(frame)

        output_array[:, :, (self.n_frames - 1) * self.img_dims[2] :] = frame

        self.buffer = output_array

        return np.expand_dims(output_array, 0)

    def normalize(self, frame):
        return self.normalization_function(frame)

    def preprocess(self, frame):
        return self.preprocess_function(frame)
