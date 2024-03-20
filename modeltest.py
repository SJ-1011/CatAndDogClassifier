import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# 모델 경로 설정
model_path = './models'

# 모델 불러오기
model = tf.keras.models.load_model(model_path)

# model.summary()

img = cv2.imread('./data/dog_test2.jpg')
resize = tf.image.resize(img, (256,256))
# plt.imshow(resize.numpy().astype(int))
# plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))

if yhat > 0.5: 
    print(f'Predicted class is Dog')
else:
    print(f'Predicted class is Cat')