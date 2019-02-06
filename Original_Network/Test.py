# import the modules we need
import cv2
import numpy as np
import random

from keras import Input, Model
from keras.engine.saving import load_model
from keras.layers import UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

from Datasets import get_image_file_names, get_im_cv2

# For tensonflow-gpu
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

# Load the model
model = load_model('./Models/weights-original-network-01-0.44.hdf5')

# Test parameters
Testing_dir = "/media/tony/MyFiles/test_256"
img_W = 256
img_H = 256


def test_images():
    color_me = get_im_cv2(get_image_file_names(Testing_dir)[1:2], 256, 256, 1)
    output = model.predict(color_me)
    print(output[0, 124, 125, 0])
    # Combine output and input to lab images
    for i in range(len(output)):
        combine = np.zeros((img_W, img_H, 3))
        combine[:, :, 0] = color_me[i][:, :, 0]
        combine[:, :, 1:] = output[i]
        combine *= 256
        combine_copy = np.uint8(combine)
        cv2.imwrite('Test_image.jpg', cv2.cvtColor(combine_copy, cv2.COLOR_LAB2BGR))


test_images();
