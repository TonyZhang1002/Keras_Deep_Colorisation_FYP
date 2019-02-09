# Load weights
import cv2
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import numpy as np

from Datasets import get_image_file_names

# Load weighs
inception = InceptionResNetV2(weights=None, include_top=True)
inception.load_weights('inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
inception.graph = tf.get_default_graph()


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

    # Read all images' and convert to Lab mode
    for path in paths:
        # Print path to debug
        print(path)

        # Read image first
        img = cv2.imread(path)
        img = cv2.resize(img, (resnet_cols, resnet_rows))

        # Switch to lab color space
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img_inRGB = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

        # Add to the imgs_class
        imgs_class.append(gray_img_inRGB)

    # Make those to array
    imgs_class = np.array(imgs_class)

    # Predict the result
    with inception.graph.as_default():
        embed = inception.predict(imgs_class)

    # Return the embedding
    return embed


get = create_inception_embedding(get_image_file_names("/Users/zhangqinyuan/Downloads/images/")[0:3])
# files_path = get_image_file_names("/Users/zhangqinyuan/Downloads/images/")[0:3]
print(get.shape)
