# import the modules we need
import os

import keras

from keras import Input, Model
from keras.layers import UpSampling2D, RepeatVector, Reshape, concatenate
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from src.Datasets import get_image_file_names, get_im_cv2



# For tensonflow-gpu
from src.Embed import create_inception_embedding
from src.Plot import training_vis

config = tf.ConfigProto(
     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
     # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

# Define the embed input shape
embed_input = Input(shape=(1000,))
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
# Fusion
fusion_output = RepeatVector(32 * 32)(embed_input)
fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3)
fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)
# Decoder
decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(fusion_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)


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
            embed_input_batch = create_inception_embedding(X_train[i:i + batch_size])
            x = images_input[:, :, :, 0]
            # Reshape the x
            x = x.reshape(x.shape + (1,))
            y = images_input[:, :, :, 1:]
            # Keep running to feed images
            yield ([x, embed_input_batch], y)


# Trainning parameters
Batch_size = 50
img_W = 256
img_H = 256
Epochs = 100
Steps_per_epoch = 3650
Val_Steps_per_epoch = 73
EarlyStopping_patience = 5
Trainning_dir = "/media/tony/MyFiles/data_256"
Validation_dir = "/media/tony/MyFiles/val_256"
Models_filepath = "./Models/weights-resnet-network-{epoch:02d}-{val_acc:.2f}.hdf5"
Trainning_file_names = get_image_file_names(Trainning_dir)
Validation_file_names = get_image_file_names(Validation_dir, 3650)

# Set the early stopping
early_stopping = EarlyStopping(monitor='val_acc', patience=EarlyStopping_patience, mode='auto')

# Set the checkpoint
checkpoint = ModelCheckpoint(Models_filepath, monitor='val_acc', verbose=1, save_best_only=True)

# Check if have any previous weight
if os.path.exists("./Models/weights-resnet-network-01-0.44.hdf5"):
    model.load_weights("./Models/weights-resnet-network-01-0.44.hdf5")
    print("Check point loaded!")

# Start trainning
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
keras.backend.get_session().run(tf.global_variables_initializer())
history = model.fit_generator(
    generator=get_train_batch(Trainning_file_names, Batch_size, img_W, img_H),
    epochs=Epochs, steps_per_epoch=Steps_per_epoch, verbose=1,
    validation_data=get_train_batch(Validation_file_names, Batch_size, img_W, img_H),
    callbacks=[checkpoint, early_stopping], validation_steps=Val_Steps_per_epoch)

# Summarize history for loss
training_vis(history)
