# import the modules we need
import numpy as np
import random

from keras import Input, Model
from keras.models import Sequential
from keras.layers import UpSampling2D, InputLayer
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10
from keras.callbacks import EarlyStopping
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from Datasets import get_image_file_names, get_im_cv2

# For tensonflow-gpu
# config = tf.ConfigProto(
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
#     # device_count = {'GPU': 1}
# )
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# set_session(session)


# Encoder
encoder_input = Input(shape=(256, 256, 1,))
encoder_output = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
# Decoder
decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
model = Model(inputs=encoder_input, outputs=decoder_output)


# Generate training data
def get_train_batch(X_train, batch_size, img_w, img_h):
    """
    :param
        X_train：List of images' path
        batch_size:
        img_w: images' width
        img_h: images' hight
    :returns
        x: A bunch of images' L layers
        y: A bunch of images' ab layers
    """
    while 1:
        for i in range(0, len(X_train), batch_size):
            images_input = get_im_cv2(X_train[i:i + batch_size], img_w, img_h, 3)
            x = images_input[:, :, :, 0]
            # Reshape the x
            x = x.reshape(x.shape + (1,))
            y = images_input[:, :, :, 1:]
            # Keep running to feed images
            yield (x, y)


# Train model
Batch_size = 2
img_W = 256
img_H = 256
Epochs = 1
Steps_per_epoch = 5
# TODO: Val and Early Stop
model.compile(optimizer='adam', loss='mse')
model.fit_generator(
    get_train_batch(get_image_file_names("/Users/zhangqinyuan/Downloads/images/"), Batch_size, img_W, img_H),
    epochs=Epochs, steps_per_epoch=Steps_per_epoch, verbose=1, workers=1)
