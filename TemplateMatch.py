import cv2
import numpy as np
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt

open_img = cv2.imread('./template/open.jpg', cv2.IMREAD_GRAYSCALE)
close_img = cv2.imread('./template/close.jpg', cv2.IMREAD_GRAYSCALE)
mask = cv2.imread('./template/mask.png', cv2.IMREAD_GRAYSCALE)

CROP_WMIN, CROP_WMAX = 700, 924
CROP_HMIN, CROP_HMAX = 326, 550
CROP_WMIN2, CROP_WMAX2 = 270, 494
CROP_HMIN2, CROP_HMAX2 = 400, 624


open_template, close_template = open_img, close_img
open_template = cv2.normalize(open_img, open_template,0,255, norm_type=cv2.NORM_MINMAX)
close_template = cv2.normalize(close_img, close_template,0,255, norm_type=cv2.NORM_MINMAX)
open_template = cv2.GaussianBlur(open_template, (5,5), 0)
close_template = cv2.GaussianBlur(close_template, (5,5), 0)
open_template, close_template = open_template.astype(np.int), close_template.astype(np.int)

def StateMatch(img):
    norm_img = 0
    #threshold = 10
    norm_img = cv2.normalize(img, norm_img,0,255, norm_type=cv2.NORM_MINMAX)
    delta_open = np.abs(norm_img - open_template)
    delta_close = np.abs(norm_img - close_template)
    count_open, count_close = np.sum(delta_open*mask), np.sum(delta_close*mask)
    if count_open > count_close:
        return 'close'
    else:
        return 'open'

def StateMatchDemo(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return StateMatch(img)

def demo():
    counter = 0
    acc = 0
    for valve_folder in os.listdir('./valve_samples'):
        for filename in os.listdir('./valve_samples/'+valve_folder):
            if os.path.splitext(filename)[1] == '.jpg':
                if(valve_folder == StateMatchDemo('./valve_samples/'+valve_folder+'/'+filename)):
                    acc += 1
            counter += 1
    print('acc = ', acc/counter, ' total = ', counter)
    
demo()
