# Recognizes-hand-gestures-and-hand-postures-used-as-video-game-controller
This is a heavily modifified version of https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe.
This including:
* Different dataset.
* Different hand gestures.
* Different hand postures.
* Modified mainly for using as an UI controller for videogames.

# How To Run
Run this using your webcam
<pre>
`python app.py`
</pre>

# Directory
<pre>
│  app.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│  
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │          
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│          
└─utils
    └─cvfpscalc.py
</pre>

# Data Labels
## Hand Posture Labels
| Index | Labels |
| :---: | :---: |
| `0` | Swipe |
| `1` | Hold |
| `2` | Pointer |
| `3` | HoldFingerTip |
## Hand Gesture Labels
| Index | Labels |
| :---: | :---: |
| `0` | Stop |
| `1` | SwipeUp |
| `2` | SwipeDown |
| `3` | SwipeLeft |
| `4` | SwipeRight |

# Model Structure
## Hand Posture Model Structure
![plot](graph/HandPostureModelStructure.png)
## Hand Gesture Model Structure
![plot](graph/HandGestureModelStructure.png)

# Model Training Results
## Hand Posture Model Training Results
### Confusion Matrix
![plot](graph/HandPostureModelConfusionMatrix.png)
### Training History
![plot](graph/HandPostureModelTrainingHistory.png)

## Hand Gesture Model Training Results
### Confusion Matrix
![plot](graph/HandGestureModelConfusionMatrix.png)
### Training History
![plot](graph/HandGestureModelTrainingHistory.png)