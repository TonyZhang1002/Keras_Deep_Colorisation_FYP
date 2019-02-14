# import the modules we need
import cv2
import keras
import numpy as np

from keras.engine.saving import load_model
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions

from src.Datasets import get_image_file_names, get_im_cv2

# For tensonflow-gpu
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

# Load the model
model = load_model('./Models/weights-resnet-network-01-0.78.hdf5')

# Load weighs
inception = InceptionResNetV2(weights=None, include_top=True)
inception.load_weights('./Models/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
inception.graph = tf.get_default_graph()
print("Resnet Weight loaded")

# Create embedding
def create_inception_embedding(paths):
    """
        :parameter：
            paths：The images' root path
        :return:
            imgs: numpy output of their classification
        """
    resnet_rows = 299
    resnet_cols = 299
    imgs_class = []

    # Read all images' and convert to gray-scale images
    for path in paths:
        # Print path to debug
        # print(path)

        # Read image first
        img = cv2.imread(path)
        img = cv2.resize(img, (resnet_cols, resnet_rows))

        # Switch to lab color space
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img_inRGB = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        # Pre-process input
        gray_img_inRGB = preprocess_input(gray_img_inRGB)

        # Add to the imgs_class
        imgs_class.append(gray_img_inRGB)

    # Make those to array
    imgs_class = np.array(imgs_class, dtype=float)

    # Predict the result
    with inception.graph.as_default():
        embed = inception.predict(imgs_class)

    label = decode_predictions(embed)
    label = label[0][0]
    print('%s (%.2f%%)' % (label[1], label[2] * 100))

    # Return the embedding
    return embed

# Test parameters
Testing_dir = "/media/tony/MyFiles/test_256"
Testing_file_names = get_image_file_names(Testing_dir)
img_W = 256
img_H = 256


def test_images():
    color_me = get_im_cv2(get_image_file_names("/media/tony/MyFiles/data_256")[200:201], img_W, img_H, 1, normalize=False, pre_processing=True)
    print(color_me[0, 250:255, 250:255, 0])
    color_me_embed = create_inception_embedding(get_image_file_names("/media/tony/MyFiles/data_256")[200:201])
    output = model.predict([color_me, color_me_embed])
    print(output[0, 250:255, 250:255, 0])
    # Combine output and input to lab images
    for i in range(len(output)):
        combine = np.zeros((img_W, img_H, 3))
        combine[:, :, 0] = color_me[i][:, :, 0]
        combine[:, :, 1:] = output[i]
        # combine *= 128
        combine_copy = np.uint8(combine)
        cv2.imwrite('Test_image.jpg', cv2.cvtColor(combine_copy, cv2.COLOR_LAB2BGR))


test_images();