#demo file for CNN on valve
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCA_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras.models import load_model
import cv2
import datetime
import numpy as np

CROP_WMIN, CROP_WMAX = 700, 924
CROP_HMIN, CROP_HMAX = 326, 550


def main():
    model = load_model('./models/MobileNetModel.h5')
    cap = cv2.VideoCapture('rtsp://admin:TURINGVIDEO123@172.16.32.8:554/Streaming/Channels/101')
    index = 0
    now = datetime.datetime.now()
    log = open('./log/'+now.strftime('log_%Y_%m_%d_%H_%M_%S')+'.txt','w')
    ret, frame = cap.read()
    while ret:
        ret, frame = cap.read()
        if(index%250 == 0):
            img = frame[np.newaxis,CROP_HMIN:CROP_HMAX,CROP_WMIN:CROP_WMAX,:]/255
            ans = model.predict(img)
            now = datetime.datetime.now()
            if ans[0,1]>ans[0,0]:
                log.write(now.strftime('%Y_%m_%d_%H_%M_%S') + " open\n")
                cv2.imwrite('./imgs/open%04d.jpg'%(index/250), frame[CROP_HMIN:CROP_HMAX,CROP_WMIN:CROP_WMAX,:])
            else:
                log.write(now.strftime('%Y_%m_%d_%H_%M_%S') + " close\n")
                cv2.imwrite('./imgs/close%04d.jpg'%(index/250), frame[CROP_HMIN:CROP_HMAX,CROP_WMIN:CROP_WMAX,:])
            if(index%2500 == 0):
                log.flush()
        index += 1
    log.close()

if __name__ == '__main__':
    main()
