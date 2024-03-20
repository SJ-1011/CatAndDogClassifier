import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import imghdr

data = tf.keras.utils.image_dataset_from_directory('./data')

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])