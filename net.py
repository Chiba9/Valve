import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Flatten, Activation, Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

NB_CLASS=3
IM_WIDTH=224
IM_HEIGHT=224
batch_size=32
EPOCH=60
train_root = './data/train'
val_root = './data/valid'
test_root = './data/test'
import matplotlib.pyplot as plt
# train data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40, # 角度值，0~180，图像旋转
    width_shift_range=0.2, # 水平平移，相对总宽度的比例
    height_shift_range=0.2, # 垂直平移，相对总高度的比例
    zoom_range=0.2, # 随机缩放范围
    horizontal_flip=True, # 一半图像水平翻转
    fill_mode='nearest' # 填充新创建像素的方法
)

train_generator = train_datagen.flow_from_directory(
    train_root,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    shuffle=True
)
train_generator.next()
#validation data
val_datagen = ImageDataGenerator(

    rescale=1./255,
    rotation_range=40,  # 角度值，0~180，图像旋转
    width_shift_range=0.2,  # 水平平移，相对总宽度的比例
    height_shift_range=0.2,  # 垂直平移，相对总高度的比例
    zoom_range=0.2,  # 随机缩放范围
    horizontal_flip=True,  # 一半图像水平翻转
    fill_mode='nearest'  # 填充新创建像素的方法
)
val_generator = val_datagen.flow_from_directory(
    val_root,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
    shuffle=True
)

# test data
test_datagen = ImageDataGenerator(
    rescale=1./255
)
test_generator = test_datagen.flow_from_directory(
    test_root,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
)

IM_WIDTH = 224
IM_HEIGHT = 224
Input_layer = Input((IM_WIDTH, IM_HEIGHT, 3))
model_inception = InceptionV3(weights = 'imagenet', include_top = False, input_shape = (IM_WIDTH, IM_HEIGHT, 3))
model = model_inception(Input_layer)
model = Flatten()(model)
model = Dense(1024)(model)
model = Dense(128)(model)
model = Dense(2)(model)
model = Activation('softmax')(model)
model = Model(Input_layer, model)
#for layer in model_inception.layers:
#    layer.trainable = False
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
best_model = ModelCheckpoint('Inception_Model.h5', verbose=1, save_best_only=True)
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])
model.summary()
train_log = model.fit_generator(
        train_generator,validation_data=val_generator,steps_per_epoch=train_generator.n/batch_size
                        ,validation_steps=val_generator.n/batch_size
                        ,epochs=EPOCH, callbacks=[TensorBoard(log_dir='mytensorboard2'),best_model, early_stop])
loss,acc=model.evaluate_generator(test_generator,steps=32)
print('Test result:loss:%f,acc:%f' % (loss, acc))


model.predict()