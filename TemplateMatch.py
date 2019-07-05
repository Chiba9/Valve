import cv2
import numpy as np
from sklearn.decomposition import PCA
open_img = cv2.imread('./template/open.jpg', cv2.IMREAD_GRAYSCALE)
close_img = cv2.imread('./template/close.jpg', cv2.IMREAD_GRAYSCALE)
CROP_WMIN, CROP_WMAX = 800, 1000
CROP_HMIN, CROP_HMAX = 300, 500

open_template, close_template = open_img[CROP_HMIN:CROP_HMAX,CROP_WMIN:CROP_WMAX], close_img[CROP_HMIN:CROP_HMAX,CROP_WMIN:CROP_WMAX]
open_template = cv2.normalize(open_img[CROP_HMIN:CROP_HMAX,CROP_WMIN:CROP_WMAX], open_template,0,255, norm_type=cv2.NORM_MINMAX)
close_template = cv2.normalize(close_img[CROP_HMIN:CROP_HMAX,CROP_WMIN:CROP_WMAX], close_template,0,255, norm_type=cv2.NORM_MINMAX)

def StateMatch(img):
    norm_img = 0
    threshold = 0.4
    cv2.normalize(img, norm_img, norm_type=cv2.NORM_MINMAX)
    delta_open = np.abs(norm_img - open_template)
    delta_close = np.abs(norm_img - close_template)
    delta_open -= np.min(delta_open)
    delta_close -= np.min(delta_close)
    count_open, count_close = np.sum(delta_open>threshold), np.sum(delta_close>threshold)
    if count_open > count_close:
        return 'close'
    else:
        return 'open'

def StateMatchDemo(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return StateMatch(img[CROP_HMIN:CROP_HMAX,CROP_WMIN:CROP_WMAX])

