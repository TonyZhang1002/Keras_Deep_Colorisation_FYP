# import the modules we need
import os

import numpy as np
import random

from keras import Input, Model
from keras.layers import UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

from Datasets import get_image_file_names, get_im_cv2

import matplotlib.pyplot as plt

# For tensonflow-gpu
config = tf.ConfigProto(
     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
     # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)


# Encoder
encoder_input = Input(shape=(256, 256, 1,))
encoder_output = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
# Decoder
decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
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


# Trainning parameters
Batch_size = 40
img_W = 256
img_H = 256
Epochs = 6
Steps_per_epoch = 45086
Val_Steps_per_epoch = 912
EarlyStopping_patience = 3
Trainning_dir = "/media/tony/MyFiles/data_256"
Validation_dir = "/media/tony/MyFiles/val_256"
Models_filepath = "./Models/weights-original-network-{epoch:02d}-{val_acc:.2f}.hdf5"
Trainning_file_names = get_image_file_names(Trainning_dir)
Validation_file_names = get_image_file_names(Validation_dir)

# Set the early stopping
early_stopping = EarlyStopping(monitor='val_acc', patience=EarlyStopping_patience, mode='auto')

# Set the checkpoint
checkpoint = ModelCheckpoint(Models_filepath, monitor='val_acc', verbose=1, save_best_only=False)

# Check if have any previous weight
if os.path.exists("./Models/weights-original-network-01-0.44.hdf5"):
    model.load_weights("./Models/weights-original-network-01-0.44.hdf5")
    print("Check point loaded!")

# Start trainning
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = model.fit_generator(
    generator=get_train_batch(Trainning_file_names, Batch_size, img_W, img_H),
    epochs=Epochs, steps_per_epoch=Steps_per_epoch, verbose=1,
    validation_data=get_train_batch(Validation_file_names, Batch_size, img_W, img_H),
    callbacks=[checkpoint], validation_steps=Val_Steps_per_epoch)

# Summarize history for loss
plt.plot(history.history["loss"])
plt.plot(history.history["val_acc"])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train","test"], loc="upper left")
plt.savefig("Performance_Dilate.jpg")

