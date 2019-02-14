# import the modules we need
import keras
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate, Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
# set_session(tf.Session(config=config))
from src.Datasets import get_image_file_names

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

Trainning_dir = "/media/tony/MyFiles/data_256"
Validation_dir = "/media/tony/MyFiles/val_256"
Models_filepath = "./Models/weights-resnet-network-{epoch:02d}-{val_acc:.2f}.h5"
Trainning_file_names = get_image_file_names(Trainning_dir)
Validation_file_names = get_image_file_names(Validation_dir, 3650)

# Get images
def getTrainningData (files_list):
    X = []
    for filename in Trainning_file_names:
        X.append(img_to_array(load_img(filename)))
    X = np.array(X, dtype=float)
    Xtrain = 1.0/255*X
    print(Xtrain.shape)
    return Xtrain


# Load weights
inception = InceptionResNetV2(weights=None, include_top=True)
inception.load_weights('../Models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
inception.graph = tf.get_default_graph()
embed_input = Input(shape=(1000,))
# Encoder
encoder_input = Input(shape=(256, 256, 1,))
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
#Fusion
fusion_output = RepeatVector(32 * 32)(embed_input)
fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3)
fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)
#Decoder
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
#Create embedding
def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed
# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.4,
        zoom_range=0.4,
        rotation_range=40,
        horizontal_flip=True)
#Generate training data
batch_size = 40
def image_a_b_gen(batch_size):
    while 1:
        for i in range(0, len(Trainning_file_names), batch_size):
            for batch in datagen.flow(getTrainningData(Trainning_file_names[i + batch_size]), batch_size=batch_size):
                grayscaled_rgb = gray2rgb(rgb2gray(batch))
                embed = create_inception_embedding(grayscaled_rgb)
                lab_batch = rgb2lab(batch)
                X_batch = lab_batch[:,:,:,0]
                X_batch = X_batch.reshape(X_batch.shape+(1,))
                Y_batch = lab_batch[:,:,:,1:] / 128
                yield ([X_batch, embed], Y_batch)


# Set the checkpoint
checkpoint = ModelCheckpoint(Models_filepath, monitor='val_acc', verbose=1, save_best_only=False)

#Train model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit_generator(image_a_b_gen(batch_size), callbacks=[checkpoint], epochs=100, steps_per_epoch=5)