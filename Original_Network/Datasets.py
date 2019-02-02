import os
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


def get_im_cv2(paths, img_rows, img_cols, color_type=1, normalize=True):
    """
    :parameter：
        paths：The images' root path
        img_rows: images' row you want
        img_cols: images' col you want
        color_type: images' output types ('L' or 'Lab')
    :return:
        imgs: numpy output
    """
    global target_img
    imgs = []

    # Read all images' and convert to Lab mode
    for path in paths:
        # Print path to debug
        # print(path)
        img = cv2.imread(path)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Reduce size
        resized = cv2.resize(lab_img, (img_cols, img_rows))
        if normalize:
            resized = resized.astype('float32')
            resized /= 256

        # Judge different output requirements
        if color_type == 1:
            target_img = resized[:, :, 0]
        elif color_type == 2:
            target_img = resized[:, :, 1:]
        elif color_type == 3:
            target_img = resized

        imgs.append(target_img)

    return np.array(imgs).reshape(len(paths), img_rows, img_cols, color_type)


# get = get_im_cv2(get_image_file_names("/Users/zhangqinyuan/Downloads/images/")[0:2], 256, 256, 3)
# print(get.shape)

# files_path = get_image_file_names("/Users/zhangqinyuan/Downloads/images/")
# print(len(files_path))
