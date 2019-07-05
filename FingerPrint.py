from keras.layers.core import Activation, Flatten, Masking, Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPool2D
from keras.layers.merge import concatenate, multiply
from keras.layers import Input
from keras.models import Model
from spline_transformer import *
import keras.backend as K
import cv2
import numpy as np

def preprocess(imgs):
    output = 0
    cv2.equalizeHist(imgs, output)
    return output

NB_Grid = 100

#N0 = 200 for free grid and 4 for PCA
def WarpEstimatorNet(x, No):
    KS_list = [32,64,64,128,256,512,1024,2048]
    for KS in KS_list:
        x = CBRM(x, KS)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(No)(x)
    return x


def CBRM(x, KS):
    C = Conv2D(filters = KS, kernel_size = (3,3), padding='same')(x)
    B = BatchNormalization()(C)
    R = Activation('relu')(B)
    M = MaxPool2D(pool_size=(2,2))(R)
    return M

def VLayer(x, KS, S):
    C = Conv2D(filters=KS,kernel_size=(3,3),strides=S, padding='same')(x)
    B = BatchNormalization()(C)
    R = Activation('relu')(B)
    return R

def DLayer(x, KS, S):
    C = Conv2DTranspose(filters=KS,kernel_size=(3,3),strides=S, padding='same')(x)
    B = BatchNormalization()(C)
    R = Activation('relu')(B)
    return R

def RigeEnhancerNet(x):
    layer1 = VLayer(x, 64, (1, 1))
    layer2 = VLayer(layer1, 128, (2, 2))
    layer3 = VLayer(layer2, 128, (1, 1))
    layer4 = VLayer(layer3, 256, (2, 2))
    layer5 = VLayer(layer4, 256, (1, 1))
    layer6 = VLayer(layer5, 512, (2, 2))
    layer7 = VLayer(layer6, 512, (1, 1))
    layer8 = VLayer(layer7, 512, (2, 2))
    layer9 = VLayer(layer8, 512, (1, 1))
    layer10 = VLayer(layer9, 512, (1, 1))
    layer11 = DLayer(layer10, 512, (2, 2))
    concat11 = concatenate([layer7, layer11], axis=3)
    layer12 = VLayer(concat11, 512, (1, 1))
    layer13 = DLayer(layer12, 256, (2, 2))
    concat13 = concatenate([layer5, layer13], axis=3)
    layer14 = VLayer(concat13, 256, (1, 1))
    layer15 = DLayer(layer14, 128, (2, 2))
    concat15 = concatenate([layer3, layer15], axis=3)
    layer16 = VLayer(concat15, 128, (1, 1))
    layer17 = DLayer(layer16, 64, (2, 2))
    concat17 = concatenate([layer1, layer17], axis=3)
    layer18 = VLayer(concat17, 2, (1,1))
    return layer18

def custom_loss_wrapper(S_tensor):
    def custom_loss(y_true, y_pred):
        return K.binary_crossentropy(y_true, y_pred) + K.mean(input_tensor)
    return custom_loss


from keras.layers import multiply
def S_norm(S_Mat):
    alpha = 0.4
    S = alpha + (1 - alpha)*(S_Mat-np.min(S_Mat))/(np.max(S_Mat)-np.min(S_Mat))
    return S

IM_WIDTH = 256
IM_HEIGHT = 256
mask = 0
from keras.utils.vis_utils import plot_model
def NetBuild():
    input_layer = Input(shape = (IM_HEIGHT, IM_WIDTH, 1))
    WEN = WarpEstimatorNet(input_layer, 2*NB_Grid)
    UnWarpedLayer = TPSTransformerLayer(WEN, control_points=NB_Grid, input_shape = (IM_HEIGHT, IM_WIDTH, 1))
    #plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=False)
    S = multiply([input_layer, input_layer])
    S = Lambda(S)
    G = RigeEnhancerNet(input_layer)
    model = Model(inputs=input_layer, outputs=G)
    model.summary()
    plot_model(model, to_file='model2.png', show_shapes=True, show_layer_names=False)

NetBuild()