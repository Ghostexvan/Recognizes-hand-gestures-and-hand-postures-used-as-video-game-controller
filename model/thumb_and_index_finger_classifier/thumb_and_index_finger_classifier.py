#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import os

full_path = os.path.realpath(__file__)
dir_path = os.path.dirname(full_path)
print("Initiate Thumb And Index Finger Gesture Classification, file at:" + os.path.dirname(full_path))

class ThumbAndIndexFingerClassifier(object):
    def __init__(
        self,
        model_path=os.path.join(dir_path,"thumb_and_index_finger_classifier.tflite"),
        score_th=0.5,
        invalid_value=0,
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(
        self,
        landmark_history,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_history], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        if np.squeeze(result)[result_index] < self.score_th:
            result_index = self.invalid_value

        return result_index
