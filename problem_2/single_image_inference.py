import os
import cv2
import numpy as np
import tensorflow as tf

import config

def get_class_name(label_number):
    if label_number == 0: return "berry"
    elif label_number == 1: return "bird"
    elif label_number == 2: return "dog"
    elif get_label_number == 3: return "flower"

def process_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img , (224, 224))
    img = img.reshape((1, 224, 224, 3))
    img = tf.keras.utils.normalize(img, axis=1)
    return np.array(img)

def get_prediction(img_path, model_path):
    # img_path = ROOT_IMG_PATH + "/" + class_name + "/" + img_name
    processed_img = process_img(img_path)
    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(processed_img)
    label_number = np.argmax(prediction)
    class_name = get_class_name(label_number)
    return class_name

if __name__ == "__main__":
    output_class_name = get_prediction(config.paths['inference_img_path'], config.paths['inference_model_path'])
    img_name = config.paths['inference_img_path'].split("/")[-1]
    print(f"The predicted label for image {img_name} is {output_class_name}")