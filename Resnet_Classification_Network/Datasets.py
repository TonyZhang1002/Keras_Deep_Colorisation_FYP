import os
import random

import numpy as np
import cv2


def get_image_file_names(root_paths):
    # Get all images' files name
    files_path = []
    for dirpath, dirnames, filenames in os.walk(root_paths):
        for file in filenames:
            if os.path.splitext(file)[1] == '.jpg':
                files_path.append(os.path.join(dirpath, file))
    return files_path


def get_im_cv2(paths, img_rows, img_cols, color_type=1, normalize=True, pre_processing = True):
    """
    :parameter：
        paths：The images' root path
        img_rows: images' row you want
        img_cols: images' col you want
        color_type: images' output types ('L' or 'Lab')
    :return:
        imgs: numpy output
    """
    global target_img, pre_processed
    imgs = []

    # Read all images' and convert to Lab mode
    for path in paths:
        # Print path to debug
        # print(path)

        # Read image first
        img = cv2.imread(path)
        img = cv2.resize(img, (img_cols, img_rows))

        # Pre-processing
        if pre_processing:
            M1 = cv2.getRotationMatrix2D((img_cols / 2, img_rows / 2), 90, 1)
            M2 = cv2.getRotationMatrix2D((img_cols / 2, img_rows / 2), 180, 1)
            pre_processing_flag = random.randint(0, 3)
            pre_processed = img
            if pre_processing_flag == 1:
                pre_processed = cv2.warpAffine(img, M1, (img_cols, img_rows))
            elif pre_processing_flag == 2:
                pre_processed = cv2.warpAffine(img, M2, (img_cols, img_rows))
            elif pre_processing_flag == 3:
                pre_processed = cv2.flip(img, -1)

        # Switch to lab color space
        lab_img = cv2.cvtColor(pre_processed, cv2.COLOR_BGR2LAB)

        # Reduce size
        if normalize:
            lab_img = lab_img.astype('float32')
            lab_img /= 256

        # Judge different output requirements
        if color_type == 1:
            target_img = lab_img[:, :, 0]
        elif color_type == 3:
            target_img = lab_img

        imgs.append(target_img)

    return np.array(imgs).reshape(len(paths), img_rows, img_cols, color_type)


# get = get_im_cv2(get_image_file_names("/media/tony/MyFiles/data_256")[0:250], 256, 256, 3)
# print(get.shape)
# print(get[1, 211, 125, 2])

# files_path = get_image_file_names("/media/tony/MyFiles/val_256")
# print(len(files_path))


