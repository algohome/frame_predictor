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

class AEMonitor(keras.callbacks.Callback):
    def __init__(self, orig_prediction_frames, outdir, offset=0, yuv=False):
        super(AEMonitor)
        self.orig_prediction_frames = orig_prediction_frames
        self.frames_to_predict = orig_prediction_frames
        self.outdir = outdir
        self.offset = offset
        if yuv:
            self.mode="YCbCr"
        else:
            self.mode="RGB"

        for i in range(5):
            frame = orig_prediction_frames[0,:,:,i*3:(i+1)*3]
            Image.fromarray((((frame+0)/1)*255).astype('uint8')).save(os.path.join(self.outdir, 'frame%04d.jpg'%(i)))

    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.frames_to_predict)[0]
        Image.fromarray((((prediction+0)/1)*255).astype('uint8'),mode=self.mode).save(os.path.join(self.outdir, 'frame%04d.jpg' % (int(epoch) + 5 + self.offset)))
        Image.fromarray((((prediction+0)/1)*255).astype('uint8'),mode=self.mode).save(os.path.join(self.outdir, 'frame.jpg'))
