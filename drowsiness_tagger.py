"""
drowsiness_tagger.py

Author: Jesus Mancilla (jesus@scrapworks.org)

Purpose: Analyze face video to export a tagged file where drowsiness was detected,

Requirements: Python 3, openCV, dlib, imutils

Usage: python drowsiness_tagger.py [shape predictor] [video file]
       [shape predictor]: For this work, the "shape_predictor_68_face_landmarks.dat" was used.
       [video file]: Video of driver face 

Code based on Adrian Rosebrock work (https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/)
"""

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import datetime
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-v", "--video", required=True,
    help="path to the video file")
args = vars(ap.parse_args())

output_name = args["video"][:-4] + '_drowsiness_tags.csv'

output = open(output_name, 'w')
output.write('time,drowsinessTag\n')

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
COUNTER = 0

print("\n[INFO] loading facial landmark predictor...\n")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

faceVid = cv2.VideoCapture(args["video"])
totalFrames = faceVid.get(cv2.CAP_PROP_FRAME_COUNT)
faceVidFPS = faceVid.get(cv2.CAP_PROP_FPS)

videoLength = totalFrames / faceVidFPS

print('Tagging', args["video"], '\n    It has a lenght of', round(videoLength/ 60, 2), 'minutes')

timePerFrame = 1/faceVidFPS
timestamp = 0
seconds = datetime.datetime.now()
oneSec = datetime.timedelta(seconds = 1)
lastFrame = 0
fps = 0

while(faceVid.isOpened()):
    timestamp += timePerFrame
    (grabbed, frame) = faceVid.read()
    currentFrame = faceVid.get(cv2.CAP_PROP_POS_FRAMES)

    if not grabbed:
        break

    if datetime.datetime.now() - seconds > oneSec:
        seconds = datetime.datetime.now()
        fps = '(' + str(currentFrame - lastFrame) + ' fps)'
        lastFrame = currentFrame

    print('    Frame', currentFrame, 'out of', totalFrames, fps, end = '\r')

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    output.write('{},{}\n'.format(timestamp, '0'))

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = np.mean([leftEAR, rightEAR])

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                output.write('{},{}\n'.format(timestamp, '1'))
        else:
            COUNTER = 0
            output.write('{},{}\n'.format(timestamp, '0'))

output.close()
faceVid.release()