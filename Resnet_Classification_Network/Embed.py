# Load weights
import time

import cv2
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2, decode_predictions, preprocess_input
import numpy as np
from keras.backend.tensorflow_backend import set_session

from Datasets import get_image_file_names

config = tf.ConfigProto(
     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
     # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

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
        print(path)

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

    # Return the embedding
    return embed


# val_names = get_image_file_names("/media/tony/MyFiles/data_256")
# get = create_inception_embedding(val_names[600:601])
# label = decode_predictions(get)
# label = label[0][0]
# print('%s (%.2f%%)' % (label[1], label[2]*100))


# start = time.clock()
#for i in range(5):
#     get = create_inception_embedding(val_names[0:40])
#     print(get.shape)
# print(time.clock() - start)
# print(get.shape)


