#demo file for CNN on valve
from keras.models import load_model
import cv2
import os
import numpy as np
import imageio

CROP_WMIN, CROP_WMAX = 700, 924
CROP_HMIN, CROP_HMAX = 326, 550


def main():
    model = load_model('./models/MobileNetModel_2.h5')
    cap = cv2.VideoCapture('rtsp://admin:TURINGVIDEO123@172.16.32.8:554/Streaming/Channels/101')
    index = 0
    ret, frame = cap.read()