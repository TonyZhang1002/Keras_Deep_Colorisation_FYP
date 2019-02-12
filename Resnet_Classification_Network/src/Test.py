# import the modules we need
import cv2
import keras
import numpy as np

from keras.engine.saving import load_model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from src.Datasets import get_image_file_names, get_im_cv2

# For tensonflow-gpu
from src.Embed import create_inception_embedding

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

# Load the model
model = load_model('./Models/weights-resnet-network-06-0.57.hdf5')

# Test parameters
Testing_dir = "/media/tony/MyFiles/test_256"
Testing_file_names = get_image_file_names(Testing_dir)
img_W = 256
img_H = 256


def test_images():
    color_me = get_im_cv2(Testing_file_names[3:4], img_W, img_H, 1, pre_processing=False)
    keras.backend.get_session().run(tf.global_variables_initializer())
    color_me_embed = create_inception_embedding(Testing_file_names[3:4])
    output = model.predict([color_me, color_me_embed])
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