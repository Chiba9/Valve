#demo file for CNN on valve
from keras.models import load_model
import cv2
import os
import numpy as np
import imageio
import numpy as np
from keras.layers import Dense, Flatten, Activation, Input
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
import tensorflow
CROP_WMIN, CROP_WMAX = 700, 924
CROP_HMIN, CROP_HMAX = 326, 550



model = load_model('./MobileNetModel_4.h5')
index = 0
ans = {}
'''
NB_CLASS=3
IM_WIDTH=224
IM_HEIGHT=224
batch_size=40
EPOCH=60
train_root = './data/train'
val_root = './data/valid'
test_root = './data/test'

test_datagen = ImageDataGenerator(
    rescale=1./255
)
test_generator = test_datagen.flow_from_directory(
    test_root,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
)
b = model.predict_generator(test_generator, steps = 10)
'''
dic = ['close','open']
for filename in os.listdir('./snap'):
    if os.path.splitext(filename)[1] == '.jpg':
        img = cv2.imread('./snap/' + filename)/255
        img = img[np.newaxis, CROP_HMIN:CROP_HMAX, CROP_WMIN:CROP_WMAX, :]
        #img = img[np.newaxis, :, :, :]
        a = np.argmax(model.predict(img))
        ans[filename] = dic[a]

print(ans)