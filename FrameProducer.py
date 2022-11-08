# Imports
import numpy as np
from PIL import Image, ImageOps
import glob
import os
import copy
import sys
import pathlib
import keras
from tensorflow.keras.layers import BatchNormalization
import cv2
from models import *
from utils import *

class FrameProducer:
    def __init__(self, model_fn, c, weights, history, data_feeder, n_frames, transitions, save_dir=None, frames_to_produce=None,mode="RGB", output_fps = 30, vid_output_name='test.avi'):
        self.model_fn = model_fn
        self.model = model_fn() # Model or model function?

        try: # This might be robust enough
            self.c_pred, self.c_true = c
            self.c = None
        except:
            self.c = c
            self.c_pred = None
            self.c_true = None

        self.weights = weights
        self.model.load_weights(self.weights)
        self.history = history
        self.data_feeder = data_feeder
        self.n_frames = n_frames # Is n_frames the number of frames to produce or predict from?
        if frames_to_produce is not None:
            self.frames_to_produce = frames_to_produce
        else:
            self.frames_to_produce = int(data_feeder.n_input_frames)-20
            #print('*'*100+'\n'+str(self.frames_to_produce))

        if save_dir is not None:
            self.save_dir = save_dir
            self.save_frames = True
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        else:
            self.save_dir = None
            self.save_frames = False



        # Validate transition types
        for transition in transitions:
            assert transition['type'] in ['weights', 'c', 'history']

        print(transitions)
        self.transitions = transitions

        #self.set_original_model()

        self.frame = 0
        if self.save_frames:
            log(os.path.join(str(self.save_dir), 'log.txt'),
                model_name = model_fn,
                c = c,
                weights = weights,
                history = history,
                transitions = transitions)
        else:
            log(vid_output_name.split('.')[0]+'_log.txt',
                model_name = model_fn,
                c = c,
                weights = weights,
                history = history,
                transitions = transitions)

        self.mode = mode

        try:
            fps = data_feeder.fps
        except:
            fps = output_fps

        self.out = cv2.VideoWriter(vid_output_name,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (self.data_feeder.img_dims[0],self.data_feeder.img_dims[1])) #mov or avi

        # Don't want these in the args list, so must hard code
        self.split_and_save_ = False
        self.split_and_save_dir = None

    def go(self):
        # Set up prediction array (need to do now to store predictions in)
        self.prediction_array = self.data_feeder.get_n_frames(self.n_frames)
        #self.split_and_show(self.prediction_array, False)

        # Begin Loop
        for i in range(self.frames_to_produce):
            if i % 10 == 0:
                print("Frame", i)
            # Predict & save next frame
            result = self.model.predict(self.prediction_array)

            if self.split_and_save_:
                self.split_and_save(self.prediction_array, denorm=False, added_frame=result[0], show=False)
            #if i > 0:
            #    self.prediction_array = holder_array

            # Save Output
            if self.save_frames:
                self.save_frame(result, False)
            self.out.write(cv2.cvtColor((result[0]*255).astype('uint8'),cv2.COLOR_RGB2BGR))

            # Set back prediction array
            self.prediction_array[0,:,:,:12] = self.prediction_array[0,:,:,3:]
            self.prediction_array[0,:,:,12:] = result # Does this go before or after?
            #self.prediction_array[0,:,:,-6:-3] = result[0]

            # Update c and history values
            self.check_transitions() # Partially implemented

            # Recover true frames
            try:
                self.true_array = self.data_feeder.get_n_frames(i+self.n_frames)#np.expand_dims(self.data_feeder.buffer,0)
            except:
                self.true_array = self.data_feeder.get_n_frames(i+self.n_frames) # Array is already preprocessed

            # Recover history
            self.prediction_array[0,:,:,:3*self.history] = self.true_array[0,:,:,:3*self.history]

            # Apply c stuff (Maybe this gets its own function)
            holder_array = copy.deepcopy(self.prediction_array)
            if self.c is not None:
                # c * true
                # 1-c * prediction
                self.prediction_array[0,:,:,12:] = (1-self.c)*result + \
                                (self.c)*(self.true_array[0,:,:,-3:])
                #print(self.frame, self.c)
            else: # 2 c vals
                self.prediction_array[0,:,:,12:] = (self.c_pred)*result + \
                                (self.c_true)*(self.true_array[0,:,:,-3:])

            self.frame += 1

        self.out.release()

    def check_transitions(self):
        # TODO: implement functionality for true_c / pred_c, with transitions
        for transition in self.transitions:
            if transition['start_frame'] < self.frame and transition['end_frame'] > self.frame:
                if transition['type'] == 'weights':
                    new_w = self.interpolate_weights(transition)
                    self.model.set_weights(new_w)

                elif transition['type'] == 'c':
                    if self.c is not None:
                        new_c = transition['transition_function'](self.frame)
                        self.c = new_c
                    else:
                        print("Multi-c transitions not yet supported")
                        #self.c_pred, self.c_true = transition['transition_function'](self.frame, transition)

                elif transition['type'] == 'history':
                    new_history = self.calculate_history(transition)
                    self.history = new_history

    def interpolate_weights(self, transition):
        temp_mod = self.model_fn()
        temp_mod.load_weights(transition['weights_1'])
        w1 = temp_mod.get_weights()
        temp_mod.load_weights(transition['weights_2'])
        w2 = temp_mod.get_weights()

        del temp_mod

        # Assume Linear for now
        p = (self.frame - transition['start_frame']) / transition['end_frame']
        #print("Weight p :", p)
        combined = []
        for i in range(len(w1)):
            c = p * w2[i] + (1 - p) * w1[i]
            combined.append(c)

        return combined

    def interpolate_c(self, transition):
        if transition['transition_function'] is None:
            # Assume linear
            # If start frame is 10, and end frame is 15, and frame is 13, we can do 13-10/15-10 for 3/5
            c = (transition['start_frame'] - self.frame) / (transition['end_frame'] - transition['start_frame'])
        else:
            c = transition['transition_function'](self.frame, transition)
        return c

    def calculate_history(self, transition):
        return transition

    def produce_frame(self, i): # What is this doing here lol
        self.prediction_array[:,:,:self.n_frames*self.history] = \
            self.true_frames[:,:,:self.n_frames*self.history]

    def save_frame(self, frame, show=False):
        if show:
            Image.fromarray((np.array(frame)*255)[0].astype('uint8'), mode=self.mode).show()
        Image.fromarray((np.array(frame)*255)[0].astype('uint8'), mode=self.mode).save(os.path.join(self.save_dir,'test_%04d.jpg'%self.frame))



    def split_and_save(self, arr, denorm=False, added_frame=None, show=False):
        assert self.split_and_save_dir is not None
        if not os.path.exists(self.split_and_save_dir):
            os.mkdir(self.split_and_save_dir)
        if added_frame is not None:
            output_array = np.ones((arr.shape[1],arr.shape[1]*6,3))
            output_array[:,-arr.shape[1]:,:] = added_frame*255
        else:
            output_array = np.ones((arr.shape[1],arr.shape[1]*5,3))

        #print(arr.shape)
        for i in range(5):
            output_array[:,i*arr.shape[1]:(i+1)*arr.shape[1],:] = arr[0,:,:,i*3:(i+1)*3]*255

        Image.fromarray(output_array.astype('uint8'), mode=self.mode).save(os.path.join(self.split_and_save_dir,'test_%04d.jpg'%self.frame))
        if show:
            Image.fromarray(output_array.astype('uint8'), mode=self.mode).show()
