import os 
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetB0, ResNet50

import config
from data_preprocess import generate_train_data, process_test_data

os.environ["CUDA_VISIBLE_DEVICES"]="0"

img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)


def simple_CNN_model():
    model = Sequential()

    model.add(InputLayer(input_shape=(256, 256, 1)))
    model.add(img_augmentation)

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(.6))  

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(.3))

    model.add(Dense(4))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.summary()

    return model


def efficientnet():

    inputs = layers.Input(shape=(224, 224, 3))
    x = img_augmentation(inputs)
    outputs = EfficientNetB0(include_top=True, weights=None, classes=4)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    model.summary()

    return model



def resnet():

    inputs = layers.Input(shape=(224, 224, 3))
    x = img_augmentation(inputs)
    outputs = tf.keras.applications.ResNet50(include_top=True, weights=None,input_tensor=None,input_shape=None,pooling=None,classes=4)(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    model.summary()

    return model

def plot_hist(history, figure_name):
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig(figure_name + '_accuracy_comparison.png')

def train_resnet_model():
    print("***Initiating Model : Resnet50")
    model = resnet()
    cp_callback = ModelCheckpoint(filepath="checkpoints/resnet_best.h5" ,save_weights_only=True, save_best_only=True, verbose=1)
    if(config.options['pre_trained']):
        model.load_weights(config.paths['pre_trained_model_path'])
    history = model.fit(train_data, train_label, 
                batch_size=config.train_params['batch_size'], 
                epochs=config.train_params['epochs'], 
                validation_data=(test_data, test_label), 
                shuffle=True, 
                callbacks=[tensorboard, cp_callback])
    model.save(config.paths["model_save_path"] + "/resnet_last.h5")
    print("***Resnet Model Training Finished***")
    plot_hist(history, figure_name="resnet")


def train_efficient_net_model():
    print("***Initiating Model : EfficientnetB0")
    model = efficientnet()
    cp_callback = ModelCheckpoint(filepath="checkpoints/efficientnet_best.h5" ,save_weights_only=True, save_best_only=True, verbose=1)
    if(config.options['pre_trained']):
        model.load_weights(config.paths['pre_trained_model_path'])
    history = model.fit(train_data, train_label, 
                batch_size=config.train_params['batch_size'], 
                epochs=config.train_params['epochs'], 
                validation_data=(test_data, test_label), 
                shuffle=True, 
                callbacks=[tensorboard, cp_callback])
    model.save(config.paths["model_save_path"] + "/efficientnet_last.h5")
    print("***Efficientnet Model Training Finished***")
    plot_hist(history, figure_name="efficientnet")


def train_model(train_mode):
    # training_mode = config.options['train_mode']
    train_data, train_label = generate_train_data(config.paths['train_data'])
    train_data_len = len(train_data)
    print(f"***Total Train Data {train_data_len}***")

    test_data, test_label = process_test_data(config.paths['test_data'])
    val_data_len = len(val_data)
    print(f"***Total Validation Data {val_data_len}***")

    tensorboard = TensorBoard(log_dir="logs/{}".format(config.paths['train_data']))

    if train_mode == "resnet":
        train_resnet_model()
        
    elif train_mode == "efficientnet":
        train_efficient_net_model()
        
    elif train_mode = "ensemble":
        train_resnet_model()
        train_efficient_net_model()
        print("***Starting the ensemble***")
        resnet_model = load_model("checkpoints/resnet_best.h5", compile=False)
        efficientnet_model = load_model("checkpoints/efficientnet_best.h5", compile=False)
        models = [keras_model, keras_model2]
        model_input = tf.keras.Input(shape=(224, 224, 3))
        model_outputs = [model(model_input) for model in models]
        ensemble_output = tf.keras.layers.Average()(model_outputs)
        ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)  
        ensemble_model.save('checkpoints/ensemble_apple.h5')
        print("finished Ensembling model")


if __name__ == "__main__":
    train_model(config.options['train_mode'])