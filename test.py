from keras.models import load_model
import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


import cv2
import numpy as np
img = cv2.imread("./valve_samples/close/0010.jpg", 0)
cv2.imwrite("canny.jpg", cv2.Canny(img, 200, 70))
cv2.imshow("canny", cv2.imread("canny.jpg"))
cv2.waitKey()
cv2.destroyAllWindows()

'''
test_root = './valve_samples (1)'

model = load_model('./ResNet_Model_6.h5')
test_datagen = ImageDataGenerator(
    rescale=1./255
)
test_generator = test_datagen.flow_from_directory(
    test_root,
    target_size=(224, 224),
    batch_size=32,
)

loss,acc = model.evaluate_generator(test_generator, steps = 32)
print('Test result:loss:%f,acc:%f' % (loss, acc))
'''