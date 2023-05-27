# Imports
import numpy as np
from PIL import Image, ImageOps
import os
import copy
from tensorflow.keras.layers import BatchNormalization
import cv2
import models as model_defs
from utils import log, make_square, check_or_make_dir, split_and_show


class FrameProducer:
    def __init__(
        self,
        model_fn,
        c,
        weights,
        history,
        data_feeder,
        n_frames,
        transitions,
        shape=(512,512),
        save_dir=None,
        frames_to_produce=None,
        mode="RGB",
        output_fps=30,
        vid_output_name="test.avi",
    ):
        self.shape = shape
        self.model_fn = model_fn
        self.model = model_fn((shape[0], shape[1], 15))

        if isinstance(c, tuple) or isinstance(c, list):  # This might be robust enough
            self.c_pred, self.c_true = c
            self.c = None
        else:
            self.c = c
            self.c_pred = None
            self.c_true = None

        self.weights = weights
        self.model.load_weights(self.weights)
        self.history = history
        self.data_feeder = data_feeder
        self.n_frames = n_frames

        if frames_to_produce is not None:
            self.frames_to_produce = frames_to_produce
        else:
            self.frames_to_produce = int(data_feeder.n_input_frames) - 20

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
            assert transition["type"] in ["weights", "c", "history"]

        self.transitions = transitions

        self.frame = 0
        if self.save_frames:
            log(
                os.path.join(str(self.save_dir), "log.txt"),
                model_name=model_fn,
                c=c,
                weights=weights,
                history=history,
                transitions=transitions,
            )
        else:
            log(
                vid_output_name.split(".")[0] + "_log.txt",
                model_name=model_fn,
                c=c,
                weights=weights,
                history=history,
                transitions=transitions,
            )

        self.mode = mode

        #FIXME: Obviously bad
        try:
            fps = data_feeder.fps
        except:
            fps = output_fps

        self.out = cv2.VideoWriter(
            vid_output_name,
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            fps,
            (self.data_feeder.img_dims[0], self.data_feeder.img_dims[1]),
        )  # mov or avi

        # Don't want these in the args list, so hard code for now (they're debugging tools iirc)
        self.split_and_save_ = False
        self.split_and_save_dir = None

    def go(self):
        # Set up prediction array (need to do now to store predictions in)
        self.prediction_array = self.data_feeder.get_n_frames(self.n_frames)

        # Begin Loop
        for i in range(self.frames_to_produce):
            if i % 10 == 0:
                print("Frame", i)
            # Predict & save next frame
            result = self.model.predict(self.prediction_array, verbose=False)

            # This is for debugging
            if self.split_and_save_:
                self.split_and_save(
                    self.prediction_array,
                    denorm=False,
                    added_frame=result[0],
                    show=False,
                )

            # Save Output
            if self.save_frames:
                self.save_frame(result, False)
            self.out.write(
                cv2.cvtColor((result[0] * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
            )

            # Set back prediction array
            self.prediction_array[0, :, :, :12] = self.prediction_array[0, :, :, 3:]
            self.prediction_array[0, :, :, 12:] = result

            # Update c and history values
            self.check_transitions()  # Partially implemented

            # Recover true frames
            self.true_array = self.data_feeder.get_n_frames(
                    i + self.n_frames
                ) 

            # Recover history
            self.prediction_array[0, :, :, : 3 * self.history] = self.true_array[
                0, :, :, : 3 * self.history
            ]

            # Apply c stuff (Maybe this gets its own function)
            if self.c is not None:
                # c * true
                # 1-c * prediction
                self.prediction_array[0, :, :, 12:] = (1 - self.c) * result + (
                    self.c
                ) * (self.true_array[0, :, :, -3:])
                # print(self.frame, self.c)
            else:  # 2 c vals
                self.prediction_array[0, :, :, 12:] = (self.c_pred) * result + (
                    self.c_true
                ) * (self.true_array[0, :, :, -3:])

            self.frame += 1

        self.out.release()

    def check_transitions(self):
        # TODO: implement functionality for true_c / pred_c, with transitions
        for transition in self.transitions:
            if (
                transition["start_frame"] < self.frame
                and transition["end_frame"] > self.frame
            ):
                if transition["type"] == "weights":
                    new_w = self.interpolate_weights(transition)
                    self.model.set_weights(new_w)

                elif transition["type"] == "c":
                    if self.c is not None:
                        new_c = transition["transition_function"](self.frame)
                        self.c = new_c
                    else:
                        print("Multi-c transitions not yet supported")
                        # self.c_pred, self.c_true = transition['transition_function'](self.frame, transition)

                elif transition["type"] == "history":
                    new_history = self.calculate_history(transition)
                    self.history = new_history

    # This seems sloppy / not very elegant.
    # TODO: Is there a way to load model weights without invoking a whole model?
    def interpolate_weights(self, transition):
        temp_mod = self.model_fn()
        temp_mod.load_weights(transition["weights_1"])
        w1 = temp_mod.get_weights()
        temp_mod.load_weights(transition["weights_2"])
        w2 = temp_mod.get_weights()

        del temp_mod

        # Assume Linear for now
        p = (self.frame - transition["start_frame"]) / transition["end_frame"]
        combined = []
        for i in range(len(w1)):
            c = p * w2[i] + (1 - p) * w1[i]
            combined.append(c)

        return combined

    def interpolate_c(self, transition):
        if not transition.get("transition_function", False):
            # Assume linear
            # If start frame is 10, and end frame is 15, and frame is 13, we can do 13-10/15-10 for 3/5
            c = (transition["start_frame"] - self.frame) / (
                transition["end_frame"] - transition["start_frame"]
            )
        else:
            c = transition["transition_function"](self.frame, transition)
        return c

    # Not Implemented
    # TODO: What was this for? A changing history value over time?
    def calculate_history(self, transition):
        return transition

    def produce_frame(self, i):  # What is this doing here lol
        self.prediction_array[:, :, : self.n_frames * self.history] = self.true_frames[
            :, :, : self.n_frames * self.history
        ]

    def save_frame(self, frame, show=False):
        if show:
            Image.fromarray(
                (np.array(frame) * 255)[0].astype("uint8"), mode=self.mode
            ).show()
        Image.fromarray(
            (np.array(frame) * 255)[0].astype("uint8"), mode=self.mode
        ).save(os.path.join(self.save_dir, "test_%04d.jpg" % self.frame))

    def split_and_save(self, arr, denorm=False, added_frame=None, show=False):
        assert self.split_and_save_dir is not None
        if not os.path.exists(self.split_and_save_dir):
            os.mkdir(self.split_and_save_dir)
        if added_frame is not None:
            output_array = np.ones((arr.shape[1], arr.shape[1] * 6, 3))
            output_array[:, -arr.shape[1] :, :] = added_frame * 255
        else:
            output_array = np.ones((arr.shape[1], arr.shape[1] * 5, 3))

        # print(arr.shape)
        for i in range(5):
            output_array[:, i * arr.shape[1] : (i + 1) * arr.shape[1], :] = (
                arr[0, :, :, i * 3 : (i + 1) * 3] * 255
            )

        Image.fromarray(output_array.astype("uint8"), mode=self.mode).save(
            os.path.join(self.split_and_save_dir, "test_%04d.jpg" % self.frame)
        )
        if show:
            Image.fromarray(output_array.astype("uint8"), mode=self.mode).show()
