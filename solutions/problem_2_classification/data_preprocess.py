import os
import cv2
import numpy as np 
import tensorflow as tf
from random import shuffle

import config

def label_image(class_name):
    """this function generates the output label from the class name

    Args:
        class_name (string): the class name of the input image

    Returns:
        int: the defined label of the class
    """    
    if class_name == "berry": return 0
    elif class_name == "bird": return 1
    elif class_name == "dog": return 2
    elif class_name == "flower": return 3

def generate_train_data(train_path):
    """the functions process the input images for model training

    Args:
        train_path (string): the path where the input images for the model is kept

    Returns:
        np.array, np.array: processed train images and their corresponding labels
    """    
    train_data = []
    train_label = []
    for class_name in os.listdir(train_path):
        for img_name in os.listdir(os.path.join(train_path, class_name)):
            # for general purpose
            # img_path = os.path.join(TRAIN_DIR, img_name)
            # for windows
            img_path = train_path + '/' + class_name + '/' + img_name 
            img = cv2.imread(img_path)
            img = cv2.resize(img , (224, 224))
            img = tf.keras.utils.normalize(img, axis=1)

            img_label = label_image(class_name)
            train_data.append(img)
            train_label.append(img_label)
    print("***Successfully Converted Train Data***")
        
    return np.array(train_data), np.array(train_label, dtype=np.uint8)

def process_test_data(test_path):
    """the functions process the input images for model training

    Args:
        test_path (string): the path where the input images for the model is kept

    Returns:
        np.array, np.array: processed test images and their corresponding labels
    """ 
    test_data = []
    test_label = []
    for class_name in os.listdir(test_path):
        for img_name in os.listdir(os.path.join(test_path, class_name)):
            # for general purpose
            # img_path = os.path.join(TRAIN_DIR, img_name)
            # for windows
            img_path = test_path + '/' + class_name + '/' + img_name 
            img = cv2.imread(img_path)
            img = cv2.resize(img , (224, 224))
            img = tf.keras.utils.normalize(img, axis=1)

            img_label = label_image(class_name)
            test_data.append(img)
            test_label.append(img_label)
    
    print("***Successfully Converted Test Data***")

    return np.array(test_data), np.array(test_label, dtype=np.uint8)

if __name__ == "__main__":
    train_data, train_label = generate_train_data(config.paths['train_data'])
    test_data, test_label = process_test_data(config.paths['test_data'])
