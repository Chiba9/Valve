import os
from PIL import Image
import imageio
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

CROP_WMIN, CROP_WMAX = 700, 924
CROP_HMIN, CROP_HMAX = 326, 550
CROP_WMIN2, CROP_WMAX2 = 270, 494
CROP_HMIN2, CROP_HMAX2 = 400, 624

def vedio_preprocess():
    for valve_folder in os.listdir('./valve_samples'):
        counter = 0
        for filename in os.listdir('./valve_samples/'+valve_folder):
            if os.path.splitext(filename)[1] == '.mp4':
                vr = imageio.get_reader('./valve_samples/'+valve_folder+'/'+filename)
                for index, im in enumerate(vr):
                    if index%30 == 0:
                        imageio.imwrite('./valve_samples/'+valve_folder+'/'+'%04d'%counter+'.jpg', im[CROP_HMIN:CROP_HMAX,CROP_WMIN:CROP_WMAX])
                        counter+=1
                        #imageio.imwrite('./valve_samples/'+valve_folder+'/'+'%04d'%counter+'.jpg', im[CROP_HMIN2:CROP_HMAX2,CROP_WMIN2:CROP_WMAX2])
                        #counter+=1
            elif os.path.splitext(filename)[1] == '.jpg':
                im = imageio.imread('./valve_samples/'+valve_folder+'/'+filename)
                imageio.imwrite('./valve_samples/' + valve_folder + '/' + '%04d' % counter + '.jpg', im)
                counter += 1

def trainsplit():
    folders1 = ['train', 'valid', 'test']
    folders2 = ['open', 'close']
    for f1 in folders1:
        path = './data/' + f1
        if not os.path.exists(path):
            os.makedirs(path)
        for f2 in folders2:
            path = './data/' + f1 + '/' + f2
            if not os.path.exists(path):
                os.makedirs(path)
    img = []
    labels = []
    for valve_folder in os.listdir('./valve_samples'):
        for filename in os.listdir('./valve_samples/'+valve_folder):
            if os.path.splitext(filename)[1] == '.jpg':
                img.append(imageio.imread('./valve_samples/'+valve_folder + '/'+filename))
                labels.append(valve_folder)
    X_train, X_val, y_train, y_val = train_test_split(img, labels, test_size=0.15)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.15/0.85)
    for i in range(0, len(y_train)):
        imageio.imwrite('./data/train/'+y_train[i] + '/%04d.jpg'%i, X_train[i])
    for i in range(0, len(y_val)):
        imageio.imwrite('./data/valid/'+y_val[i] + '/%04d.jpg'%i, X_val[i])
    for i in range(0, len(y_test)):
        imageio.imwrite('./data/test/'+y_test[i] + '/%04d.jpg'%i, X_test[i])

vedio_preprocess()
trainsplit()