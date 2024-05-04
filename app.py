#!/usr/bin/env python
# -*- coding: utf-8 -*-

# IMPORT LIBRARIES USED ---------------------------------------------------------------------------------------
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import os

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc

from model import KeyPointClassifier
from model import PointHistoryClassifier
from model import HandGestureClassifier
from model import ThumbAndIndexFingerClassifier

# SPECIFY THE PATH USED ---------------------------------------------------------------------------------------

## Path to this file
full_path = os.path.realpath(__file__)

## Path to the directory containing this file
dir_path = os.path.dirname(full_path)

# PASS THE NECESSARY ARGUMENTS --------------------------------------------------------------------------------
def get_args():
    ## This function returns the necessary arguments for the program
    ## --------------------------------------------------------------------------------------------------------
    
    ## Variable refers to the object used to pass arguments
    parser = argparse.ArgumentParser()

    ## Arguments for OpenCV:
    ### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ### Device index used to capture the video
    parser.add_argument("--device", type=int, default=0)
    
    ### Width of the captured video
    parser.add_argument("--width", help='cap width', type=int, default=1600)
    
    ### Height of the captured video
    parser.add_argument("--height", help='cap height', type=int, default=900)
    ### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ## Arguments for hand recognition from MediaPipe:
    ### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ### Whether to use static image from input or not
    #### ------------------------------------------------------------------------------------------------------
    #### If set to FALSE, the solution treats the input images as a video stream and will try to detect hands 
    #### in the first input images, and upon a successful detection further localizes the hand landmarks. In 
    #### subsequent images, once all max_num_hands hands are detected and the corresponding hand landmarks are
    #### localized, it simply tracks those landmarks without invoking another detection until it loses track of
    #### any of the hands. This reduces latency and is ideal for processing video frames.
    #### ------------------------------------------------------------------------------------------------------
    #### If set to TRUE, hand detection runs on every input image, ideal for processing a batch of static, 
    #### possibly unrelated, images.
    parser.add_argument('--use_static_image_mode', action='store_true')
    
    ### Minimum dectection confidence value 
    #### ------------------------------------------------------------------------------------------------------
    #### The minimum confidence value (from 0.0 to 1.0) from the hand detection model for the detection to be 
    #### considered successful.
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    
    ### Minimum tracking confidence value
    #### ------------------------------------------------------------------------------------------------------
    #### Minimum confidence value (from 0.0 to 1.0) from the landmark-tracking model for the hand landmarks to
    #### be considered tracked successfully, or otherwise hand detection will be invoked automatically on the 
    #### next input image. Setting it to a higher value can increase robustness of the solution, at the expense
    #### of a higher latency. 
    #### ------------------------------------------------------------------------------------------------------
    #### Ignored if static_image_mode is true, where hand detection simply runs on every image.
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    ### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    ## Convert argument strings to objects and assign them as attributes of the namespace
    args = parser.parse_args()

    return args

# MAIN PART OF THIS PROGRAM -----------------------------------------------------------------------------------
def main():
    ## This is the main flow of the program 
    ## --------------------------------------------------------------------------------------------------------
    
    ## Receive object containing the arguments
    args = get_args()

    ## Get arguments from the object containing the arguments:
    ### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ### For OpenCV
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    ### For MediaPipe
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    ### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ## Whether to draw a bounding rectangular or not
    use_brect = True

    ## Set up the camera and set the necessary parameters
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    ## Set up the hand recognition model and set the necessary parameters
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1, ### Maximum number of recognized hands
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    ## Set up the hand posture classification model
    keypoint_classifier = KeyPointClassifier()

    ## Set up the index finger gesture classification model
    point_history_classifier = PointHistoryClassifier()
    
    ## Set up the hand gesture classification model
    hand_gesture_classifier = HandGestureClassifier()
    
    ## Set up the thumb and index finger gesture classification model
    thumb_and_index_finger_classifier = ThumbAndIndexFingerClassifier()

    ## Get labels for classification:
    ### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    ### For hand posture classification
    with open(os.path.join(dir_path, "model/keypoint_classifier/keypoint_classifier_label.csv"),
            encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    
    ### For index finger gesture classification
    with open(os.path.join(dir_path, "model/point_history_classifier/point_history_classifier_label.csv"),
              encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]
        
    ### For hand gesture classification
    with open(os.path.join(dir_path, "model/hand_gesture_classifier/hand_gesture_classifier_label.csv"),
              encoding='utf-8-sig') as f:
        hand_gesture_classifier_labels = csv.reader(f)
        hand_gesture_classifier_labels = [
            row[0] for row in hand_gesture_classifier_labels
        ]
        
    ### For thumb and index finger gesture classification
    with open(os.path.join(dir_path, "model/thumb_and_index_finger_classifier/thumb_and_index_finger_classifier_label.csv"),
              encoding='utf-8-sig') as f:
        thumb_and_index_finger_classifier_labels = csv.reader(f)
        thumb_and_index_finger_classifier_labels = [
            row[0] for row in thumb_and_index_finger_classifier_labels
        ]    
    ### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    ## Display the camera's FPS
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    ## Point history list
    history_length = 16
    point_history = deque(maxlen=history_length)
    
    ## Landmark history list
    landmark_history = deque(maxlen=history_length)
    
    ## Thumb and index finger history list
    thumb_and_index_finger_history = deque(maxlen=history_length)

    ## Finger gesture classification history length
    finger_gesture_history = deque(maxlen=history_length)
    
    ## Hand gesture classification history length
    hand_gesture_history = deque(maxlen=history_length)
    
    ## Thumb and index finger classification history length
    thumb_and_index_finger_gesture_history = deque(maxlen=history_length)

    ## Program mode
    mode = 0

    ## Main loop of the program
    while True:
        ### Get the FPS of the camera
        fps = cvFpsCalc.get()

        ### Display the window until a key to be pressed
        key = cv.waitKey(10)
        
        ### If the key pressed is ESC
        if key == 27: #### Key Code for ESC
            #### Exit the program
            break
        
        ### Set the program mode based on the key pressed
        number, mode = select_mode(key, mode)

        ### Grabs, decodes and returns the next video frame
        ret, image = cap.read()
        
        ### If nothing is retrieved
        if not ret:
            #### Exit the program
            break
        
        ### Mirror flip for display
        image = cv.flip(image, 1) 
        
        ### Deep copy the input image
        debug_image = copy.deepcopy(image)

        ### Convert image from BGR to RGB color system
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        ### Lock the image data
        image.flags.writeable = False
        
        ### Proceed to identify the landmarks of the hand
        results = hands.process(image)
        
        ### Unlock the image data
        image.flags.writeable = True
        
        hand_sign_id=-1

        ### If there are results of hand landmarks recognition
        if results.multi_hand_landmarks is not None:
            #### With hand landmarks corresponding to the hand direction
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                ##### Calculate hand boundaries
                brect = calc_bounding_rect(
                    debug_image, 
                    hand_landmarks
                    )
                
                ##### Calculate hand landmarks
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                ##### Preprocess the obtained data
                ###### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                ###### Hand landmark data
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list
                )
                
                ###### Index finger history data
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                
                ###### Hand landmark history data
                pre_processed_hand_landmark_history_list = pre_process_hand_landmark_history(
                    debug_image, landmark_history)
                
                ###### Thumb and index finger history data
                pre_processed_thumb_and_index_finger_history_list = pre_process_thumb_and_index_finger_history(
                    debug_image, thumb_and_index_finger_history
                )
                ###### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                
                # ##### Store study data
                # logging_csv(number, 
                #             mode, 
                #             pre_processed_landmark_list,
                #             pre_processed_point_history_list)

                ##### Classification of hand posture
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                ##### Store study data
                logging_csv(number, 
                            mode, 
                            pre_processed_landmark_list,
                            pre_processed_point_history_list,
                            pre_processed_hand_landmark_history_list,
                            pre_processed_thumb_and_index_finger_history_list)
                
                ##### Save index finger coordinate data
                point_history.append(landmark_list[8])
                
                ##### Save landmark history data
                # landmark_history.append(landmark_list)
                landmark_history.append([
                                            landmark_list[0], 
                                            landmark_list[4], 
                                            landmark_list[8], 
                                            landmark_list[12],
                                            landmark_list[16],
                                            landmark_list[20]
                                        ])
                
                ##### Save thumb and index finger history data
                thumb_and_index_finger_history.append([landmark_list[4], landmark_list[8]])

                ##### Assign a recognized index finger gesture default ID
                finger_gesture_id = 0
                
                ##### Assign a recognized hand gesture default ID
                hand_gesture_id = 0
                
                ##### Assign a recognized thumb and index finger gesture default ID
                thumb_and_index_finger_id = 0
                
                ##### Get the length of the history of index finger data
                point_history_len = len(pre_processed_point_history_list)
                
                ##### Get the length of the history of landmark data
                landmark_history_len = len(pre_processed_hand_landmark_history_list)
                
                ##### Get the length of the thumb and index finger data
                thumb_and_index_finger_history_len = len(pre_processed_thumb_and_index_finger_history_list)
                
                ##### If the historical length is long enough
                if point_history_len == (history_length * 2):
                    ###### Classification of index finger gestures
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)
                    
                if landmark_history_len == history_length*12:
                    ###### Classification of index finger gestures
                    hand_gesture_id = hand_gesture_classifier(
                        pre_processed_hand_landmark_history_list)
                    
                if thumb_and_index_finger_history_len == history_length*4:
                    ###### Classification of thumb and index finger gestures
                    thumb_and_index_finger_id = thumb_and_index_finger_classifier(
                        pre_processed_thumb_and_index_finger_history_list)

                ##### Save the index finger gesture ID that appear
                finger_gesture_history.append(finger_gesture_id)
                print(finger_gesture_history)
                
                ##### Save the hand gesture ID that appear
                hand_gesture_history.append(hand_gesture_id)
                
                ##### Save the thumb and index finger gesture ID that appear
                thumb_and_index_finger_gesture_history.append(thumb_and_index_finger_id)
                
                ##### Get the most frequently occurring index finger gesture IDs
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()
                
                ##### Get the most frequently occuring hand gesture IDs
                most_common_hg_id = Counter(
                    hand_gesture_history).most_common()
                
                ##### Get the most frequently occuring thumb and index gesture IDs
                most_common_tig_id = Counter(
                    thumb_and_index_finger_gesture_history).most_common()

                ##### Draw a border around the recognized hand
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                
                ##### Draw hand landmarks on the recognized hand
                debug_image = draw_landmarks(debug_image, landmark_list)
                
                ##### Draw text information on the screen
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                    hand_gesture_classifier_labels[most_common_hg_id[0][0]],
                    thumb_and_index_finger_classifier_labels[most_common_tig_id[0][0]],
                    str(len(landmark_list)),
                    str(str(landmark_list[8]))
                )
        ### If there are no hand landmark identification results
        else:
            #### Save empty index finger data to history
            point_history.append([0, 0])
            
            #### Save empty thumb and index finger data to history
            thumb_and_index_finger_history.append([[0, 0], [0, 0]])
            landmark_history.append([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])

        ### Draw the history of hand landmarks
        if hand_sign_id==2 or mode>=3:
            # debug_image = draw_point_history(debug_image, point_history)
            debug_image = draw_landmark_history(debug_image, landmark_history)
        
        ### Draw information on the screen
        debug_image = draw_info(debug_image, fps, mode, number)

        ### Display results on the screen
        cv.imshow('Index Finger Gesture Recognition', debug_image)

    ## Turn off the camera
    cap.release()
    
    ## Turn off the program window
    cv.destroyAllWindows()

# SELECT THE RUN MODE FOR THE PROGRAM -------------------------------------------------------------------------
def select_mode(key, mode):
    ## This function returns the program's running mode as well as the number used to store data (if any)
    ## --------------------------------------------------------------------------------------------------------
    
    ## Delete the number used to save data
    number = -1
    
    ## With number keys from 1 to 9
    if 48 <= key <= 57:  ### Key Code for number keys from 1 to 9
        ### Get the number used to save the corresponding data
        number = key - 48
        
    ## Set the program's running mode with the corresponding keys
    ### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    if key == 110:  #### Key Code for N
        #### The program's default running mode only performs identification and classification
        mode = 0
    
    if key == 107:  #### Key Code for K
        #### Hand posture data recording mode, recorded with the number corresponding to each hand posture
        mode = 1
        
    if key == 104:  #### Key Code for H
        #### Index finger gesture data recording mode, recorded with the number corresponding to each index 
        #### finger gesture
        mode = 2
        
    if key == 106:  #### Key Code for J
        #### Hand gesture data recording mode, recorded with the number corresponding to each hand gesture
        mode = 3
    
    if key == 108:  #### Key Code for L
        ##### Thumb and index finger data recording mode, recorded with the number corresponding to each thumb 
        ##### and index finger gesture
        mode = 4
    
    ### >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
    return number, mode

# CALCULATE THE BOUNDARY AROUNG THE RECOGNIZED HAND -----------------------------------------------------------
def calc_bounding_rect(image, landmarks):
    ## This function returns the coordinates of the four vertices of the rectangle surrounding  
    ## the detected hand
    ## --------------------------------------------------------------------------------------------------------

    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

# CALCULATE THE LIST OF HAND LANDMARKS ------------------------------------------------------------------------
def calc_landmark_list(image, landmarks):
    ## This function returns a list of landmarks on the hand
    ## --------------------------------------------------------------------------------------------------------
    
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    ## Keypoints
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

# PREPROCESSING THE HAND LANDMARK DATA
def pre_process_landmark(landmark_list):
    ## This function returns the hand landmark data after preprocessing
    ## --------------------------------------------------------------------------------------------------------
    
    temp_landmark_list = copy.deepcopy(landmark_list)
    # print("[INFO] Length of landmark list before list: ", len(temp_landmark_list), " with shape: ", np.array(temp_landmark_list).shape)

    ## Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    ## Convert to 1D list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))
    # print("[INFO] Length of landmark list after chain: ", len(temp_landmark_list), " with shape: ", np.array(temp_landmark_list).shape)

    ## Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    # print("[INFO] Length of landmark list after map: ", len(temp_landmark_list), " with shape: ", np.array(temp_landmark_list).shape)

    return temp_landmark_list

# PREPROCESS DATA RECORDED ACCORDING TO HAND LANDMARK HISTORY -------------------------------------------------
def pre_process_point_history(image, point_history):
    ## This function returns hand landmark historical data after preprocessing
    ## --------------------------------------------------------------------------------------------------------
    
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    ## Convert to relative coordinates
    
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            currentX, currentY = point[0], point[1]
            base_x, base_y = point[0], point[1]

        if (abs(temp_point_history[index][0] - currentX) < 20):
            temp_point_history[index][0] = currentX
        else:
            currentX = temp_point_history[index][0]
            
        if (abs(temp_point_history[index][1] - currentY) < 20):
            temp_point_history[index][1] = currentY
        else:
            currentY = temp_point_history[index][1]
        
        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    ## Convert to 1D list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def pre_process_hand_landmark_history(image, hand_landmark_history):
    ## This function returns 
    ## --------------------------------------------------------------------------------------------------------
    
    image_width, image_height = image.shape[1], image.shape[0]

    temp_hand_landmark_history = copy.deepcopy(hand_landmark_history)
    
    base = [[None for _ in range(2)] for _ in range(6)]

    ## Convert to relative coordinates
    for index, point in enumerate(temp_hand_landmark_history):
        if index == 0:
            for _index, _point in enumerate(point):
                base[_index][0], base[_index][1] = _point[0], _point[1]
                
        for _index, _point in enumerate(point):
                temp_hand_landmark_history[index][_index][0] = (temp_hand_landmark_history[index][_index][0] - 
                                                                base[_index][0]) / image_width
                temp_hand_landmark_history[index][_index][1] = (temp_hand_landmark_history[index][_index][1] - 
                                                                base[_index][1]) / image_height

    ## Convert to 1D list
    temp_hand_landmark_history = list(
        itertools.chain.from_iterable(temp_hand_landmark_history))
    
    temp_hand_landmark_history = list(
        itertools.chain.from_iterable(temp_hand_landmark_history))

    return temp_hand_landmark_history

def pre_process_thumb_and_index_finger_history(image, thumb_and_index_finger_history):
    ## This function returns 
    ## --------------------------------------------------------------------------------------------------------
    
    image_width, image_height = image.shape[1], image.shape[0]

    temp_thumb_and_index_finger_history = copy.deepcopy(thumb_and_index_finger_history)
    # print(temp_thumb_and_index_finger_history[0][0][0])

    ## Convert to relative coordinates
    thumb_base_x, thumb_base_y = 0, 0
    index_base_x, index_base_y = 0, 0
    for index, point in enumerate(temp_thumb_and_index_finger_history):
        
        if index == 0:
            # print (point)
            thumb_base_x, thumb_base_y = point[0][0], point[0][1]
            index_base_x, index_base_y = point[1][0], point[1][1]

        temp_thumb_and_index_finger_history[index][0][0] = (temp_thumb_and_index_finger_history[index][0][0] -
                                                            thumb_base_x) / image_width
        temp_thumb_and_index_finger_history[index][0][1] = (temp_thumb_and_index_finger_history[index][0][1] -
                                                            thumb_base_y) / image_height
        
        temp_thumb_and_index_finger_history[index][1][0] = (temp_thumb_and_index_finger_history[index][1][0] -
                                                            index_base_x) / image_width
        temp_thumb_and_index_finger_history[index][1][1] = (temp_thumb_and_index_finger_history[index][1][1] -
                                                            index_base_y) / image_height

    ## Convert to 1D list
    temp_thumb_and_index_finger_history = list(
        itertools.chain.from_iterable(temp_thumb_and_index_finger_history))
    
    temp_thumb_and_index_finger_history = list(
        itertools.chain.from_iterable(temp_thumb_and_index_finger_history))

    return temp_thumb_and_index_finger_history

# WRITE DATA TO CSV FILE --------------------------------------------------------------------------------------
def logging_csv(number, mode, landmark_list, point_history_list, landmark_history_list, thumb_and_index_finger_history_list):
    ## This function writes the collected data to the path corresponding to the program mode
    ## --------------------------------------------------------------------------------------------------------
    
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = os.path.join(dir_path, 'model/keypoint_classifier/keypoint.csv')
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = os.path.join(dir_path,'model/point_history_classifier/point_history.csv')
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    if mode == 3 and (0 <= number <= 9):
        csv_path = os.path.join(dir_path,'model/hand_gesture_classifier/hand_gesture.csv')
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_history_list])
    if mode == 4 and (0 <= number <= 9):
        csv_path = os.path.join(dir_path,'model/thumb_and_index_finger_classifier/thumb_and_index_finger.csv')
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *thumb_and_index_finger_history_list])        
    return

# DRAW HAND LANDMARKS ON THE SCREEN ---------------------------------------------------------------------------
def draw_landmarks(image, landmark_point):
    ## This function draws landmarks on the hand onto the display screen
    ## --------------------------------------------------------------------------------------------------------
    
    ## Connection line
    if len(landmark_point) > 0:
        ### Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        ### Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        ### Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        ### Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        ### Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        ### Palm of the hand
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    ## Keypoints
    for index, landmark in enumerate(landmark_point):
        if index == 0:  ### Wrist 1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  ### Writst 2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            
        if index == 2:  ### Thumb: Base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  ### Thumb: 1st joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  ### Thumb: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            
        if index == 5:  ### Index finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  ### Index finger: 2nd joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  ### Index finger: 1st joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  ### Index finger: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            
        if index == 9:  ### Middle finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  ### Middle finger: 2nd joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  ### Middle finger: 1st joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  ### Middle finger: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            
        if index == 13:  ### Ring finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  ### Ring finger: 2nd joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  ### Ring finger: 1st joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  ### Ring finger: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            
        if index == 17:  ### Little finger: base
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  ### Little finger: 2nd joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  ### Little finger: 1st joint
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  ### Little finger: fingertip
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

# DRAW A BORDER AROUND THE DETECTED HAND ----------------------------------------------------------------------
def draw_bounding_rect(use_brect, image, brect):
    ## This function draws a border around the detected hand
    ## --------------------------------------------------------------------------------------------------------
    
    if use_brect:
        ### Circumscribed rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

# DRAW TEXT INFORMATION ON THE DISPLAY SCREEN -----------------------------------------------------------------
def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text, hand_gesture_text, thumb_and_index_finger_gesture_text, landmark, index_coordinate):
    ## This function draws text information on the display screen
    ## --------------------------------------------------------------------------------------------------------
    
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Index Finger Gesture:" + finger_gesture_text, (10, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Index Finger Gesture:" + finger_gesture_text, (10, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)
        # cv.putText(image, landmark, (10, 110),
        #            cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        # cv.putText(image, landmark, (10, 110),
        #            cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
        #            cv.LINE_AA)
        # cv.putText(image, index_coordinate, (10, 150),
        #            cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        # cv.putText(image, index_coordinate, (10, 150),
        #            cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
        #            cv.LINE_AA)
        
    if hand_gesture_text != "":
        cv.putText(image, "Hand Gesture:" + hand_gesture_text, (10, 110),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Hand Gesture:" + hand_gesture_text, (10, 110),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)
    
    # if thumb_and_index_finger_gesture_text != "":
    #     cv.putText(image, "Thumb and Index Finger Gesture:" + thumb_and_index_finger_gesture_text, (10, 150),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    #     cv.putText(image, "Thumb and Index Finger Gesture:" + thumb_and_index_finger_gesture_text, (10, 150),
    #                cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
    #                cv.LINE_AA)
        
    return image

# DRAW THE HISTORY OF HAND LANDMARKS ON THE DISPLAY SCREEN ----------------------------------------------------
def draw_point_history(image, point_history):
    ## This function draws the history of hand landmarks on the display screen
    ## --------------------------------------------------------------------------------------------------------
    
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image

# DRAW THE HISTORY OF HAND LANDMARKS ON THE DISPLAY SCREEN ----------------------------------------------------
def draw_landmark_history(image, landmark_history):
    ## This function draws the history of hand landmarks on the display screen
    ## --------------------------------------------------------------------------------------------------------
    
    for index, point in enumerate(landmark_history):
        for _index, _point in enumerate(point):
            if _point[0] != 0 and _point[1] != 0:
                match _index:
                    # case 0:
                    #     cv.circle(image, (_point[0], _point[1]), 1 + int(index / 2),
                    #             (34, 139, 34), 2)
                        # break
                    # case 1:
                    #     cv.circle(image, (_point[0], _point[1]), 1 + int(index / 2),
                    #             (255, 0, 0), 2)
                    #     # break
                    case 2:
                        cv.circle(image, (_point[0], _point[1]), 1 + int(index / 2),
                                (0, 0, 255), 2)
                        # break
                    # case 3:
                    #     cv.circle(image, (_point[0], _point[1]), 1 + int(index / 2),
                    #             (0, 255, 0), 2)
                    #     # break
                    # case 4:
                    #     cv.circle(image, (_point[0], _point[1]), 1 + int(index / 2),
                    #             (224, 255, 255), 2)
                    #     # break
                    # case 5:
                    #     cv.circle(image, (_point[0], _point[1]), 1 + int(index / 2),
                    #             (255, 255, 0), 2)
                    #     # break
    return image

# DRAW INFORMATION ON THE DISPLAY SCREEN ----------------------------------------------------------------------
def draw_info(image, fps, mode, number):
    ## This function draws information on the display screen
    ## --------------------------------------------------------------------------------------------------------
    
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Hand Posture', 'Logging Index Finger Gesture', 'Logging Hand Gesture', 'Logging Thumb And Index Finger Gesture']
    if 1 <= mode <= 4:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, image.shape[0] - 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, image.shape[0] - 40),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image

# THE MAIN PROGRAM RUNS HERE ----------------------------------------------------------------------------------
if __name__ == '__main__':
    main()