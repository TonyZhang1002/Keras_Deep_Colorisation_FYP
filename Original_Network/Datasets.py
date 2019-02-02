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
    imgs = []

    # Read all images' and convert to Lab mode
    for path in paths:
        print(path)
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
        elif color_type == 3:
            target_img = resized

        imgs.append(target_img)

    return np.array(imgs).reshape(len(paths), img_rows, img_cols, color_type)


def get_train_batch(X_train, y_train, batch_size, img_w, img_h, color_type):
    """
    :param：
        X_train：所有图片路径列表
        y_train: 所有图片对应的标签列表
        batch_size:批次
        img_w:图片宽
        img_h:图片高
        color_type:图片类型
        is_argumentation:是否需要数据增强
    返回:
        一个generator，x: 获取的批次图片 y: 获取的图片对应的标签
    """
    while 1:
        for i in range(0, len(X_train), batch_size):
            x = get_im_cv2(X_train[i:i + batch_size], img_w, img_h, color_type)
            y = y_train[i:i + batch_size]
            # 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
            yield ({'input': x}, {'output': y})


get = get_im_cv2(get_image_file_names("/Users/zhangqinyuan/Downloads/images/")[0:2], 256, 256, 3)
print(get.shape)

# files_path = get_image_file_names("/Users/zhangqinyuan/Downloads/images/")[0:2]
# print(files_path)
