import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from data_preprocess import generate_train_data

import config

# training image directory
TRAIN_DIR = "dataset_256X256/train"
# testing image directory
TEST_DIR = "dataset_256X256/test"

def check_img_size(check_dir):
    for class_name in os.listdir(check_dir):
        for img_name in os.listdir(os.path.join(check_dir, class_name)):
            # for general purpose
            # img_path = os.path.join(TRAIN_DIR, img_name)
            # for windows
            img_path = check_dir + '/' + class_name + '/' + img_name 
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # print(img.shape)
            if img.shape != (256, 256):
                return "size missmatch"
    return "all are same size"
            



if __name__ == "__main__":
    # train_data = generate_train_data()
    # plt.imshow(train_data[0][0])
    # plt.show()
    # print(check_img_size(TRAIN_DIR))
    # print(check_img_size(TEST_DIR))
    # for class_name in os.listdir(TRAIN_DIR):
    #     for img_name in os.listdir(os.path.join(TRAIN_DIR, class_name)):
    #         # for general purpose
    #         # img_path = os.path.join(TRAIN_DIR, img_name)
    #         # for windows
    #         img_path = TRAIN_DIR + '/' + class_name + '/' + img_name 
    #         img = cv2.imread(img_path)
    #         img = cv2.resize(img, (224, 224))
    #         print(img.shape)
    #         img = tf.keras.utils.normalize(img, axis=1)
    #         print(img)
    #         # plt.imshow(img)
    #         # plt.show()
    #         break
    print(config.paths['train_data'])
