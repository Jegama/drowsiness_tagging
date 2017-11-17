# Drowsiness tagger

This script analyzes a face video looking at eye gestures in order to export a tagged file where drowsiness was detected.

## Dependecies

- Python 3.6+
- OpenCV
- dlib
- imutils

## How to run

`python drowsiness_tagger.py [shape predictor] [video file]`

- [shape predictor]: For this work, the `shape_predictor_68_face_landmarks.dat` was used.
- [video file]: Video of driver's face 

Code based on [Adrian Rosebrock](https://github.com/jrosebr1)'s [tutorial](https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)