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

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
    """This function defines a simple CNN model

    Returns:
        tf.keras.model: return the defined model
    """ 
    # initiate a sequential model
    model = Sequential()
    # define input layer
    model.add(InputLayer(input_shape=(256, 256, 1)))
    # add augmentation layer
    model.add(img_augmentation)

    # define convulation layer, activation function, max pooling layer and batch normalization
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    # redefine convulation layer, activation function, max pooling layer and batch normalization
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    # flatten the CNN output
    model.add(Flatten())
    model.add(Dropout(.6))  

    # define dense layer
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(.3))

    # output layer
    model.add(Dense(4))
    model.add(Activation('softmax'))

    # compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    # print model summary
    model.summary()

    return model


def efficientnet():
    """This function defines the backbone of a efficientnet model

    Returns:
        tf.keras.model: return the defined model
    """ 
    # define input layer
    inputs = layers.Input(shape=(224, 224, 3))
    # add augmentation layer
    x = img_augmentation(inputs)
    # define efficientnet config
    outputs = EfficientNetB0(include_top=True, weights=None, classes=4)(x)
    # generate the effcienet model
    model = tf.keras.Model(inputs, outputs)
    # compile the model
    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    # print model summary
    model.summary()

    return model



def resnet():
    """This function defines the backbone of a resnet model

    Returns:
        tf.keras.model: return the defined model
    """    
    # define input shape
    inputs = layers.Input(shape=(224, 224, 3))
    # add augmentation layer
    x = img_augmentation(inputs)
    # define resnet config
    outputs = tf.keras.applications.ResNet50(include_top=True, weights=None,input_tensor=None,input_shape=None,pooling=None,classes=4)(x)

    # generate the resnet model
    model = tf.keras.Model(inputs, outputs)

    # compile the model
    model.compile(
        optimizer="adam", 
        loss="sparse_categorical_crossentropy", 
        metrics=["accuracy"]
    )
    # print model summary
    model.summary()

    return model

def plot_hist(history, figure_name):
    """ This function plot the accuracy given history of a model

    Args:
        history (object of dicionary): the history of a model
        figure_name (string): the suffix of the name of figure which willl show the accuracy
    """    
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.savefig("logs/" + figure_name + '_accuracy_comparison' + config.options['logs_name'] + '.png')
    # clear plt buffer
    plt.clf()

def train_resnet_model(train_data, train_label, test_data, test_label, tensorboard, pre_trained):
    """This functions trains the resnet model based on given parameters

    Args:
        train_data (list): the np array of training data
        train_label (list): the np array of training labels
        test_data (list): the np array of validation data
        test_label (list): the bp array of validation labels
        tensorboard (callback): tensorboard log callback
        pre_trained (bool): this flag defines whether to use pre trained model or not
    """ 
    print("***Initiating Model : Resnet50")
    model = resnet()
    # define checkpoint for saving the best model
    cp_callback = ModelCheckpoint(filepath='checkpoints/resnet_best_' + config.options['logs_name'] + '.h5' ,save_weights_only=False, save_best_only=True, verbose=1)
    # load pre-trained model
    if(pre_trained):
        model.load_weights(config.paths['pre_trained_model_path'])
    # train the model
    history_resnet = model.fit(train_data, train_label, 
                batch_size=config.train_params['batch_size'], 
                epochs=config.train_params['epochs'], 
                validation_data=(test_data, test_label), 
                shuffle=True, 
                callbacks=[tensorboard, cp_callback])
    # save the model of the last epoch
    model.save("models/resnet_last_" + config.options['logs_name'] + ".h5")
    print("***Resnet Model Training Finished***")
    plot_hist(history_resnet, figure_name="resnet")


def train_efficient_net_model(train_data, train_label, test_data, test_label, tensorboard, pre_trained):
    """This functions trains the efficientnet model based on given parameters

    Args:
        train_data (list): the np array of training data
        train_label (list): the np array of training labels
        test_data (list): the np array of validation data
        test_label (list): the bp array of validation labels
        tensorboard (callback): tensorboard log callback
        pre_trained (bool): this flag defines whether to use pre trained model or not
    """    
    print("***Initiating Model : EfficientnetB0")
    model = efficientnet()
    # define checkpoint for saving the best model
    cp_callback = ModelCheckpoint(filepath='checkpoints/efficientnet_best_' + config.options['logs_name'] + '.h5' ,save_weights_only=False, save_best_only=True, verbose=1)

    # load pre trained data
    if(pre_trained):
        model.load_weights(config.paths['pre_trained_model_path'])
    
    # train the model
    history_efficientnet = model.fit(train_data, train_label, 
                batch_size=config.train_params['batch_size'], 
                epochs=config.train_params['epochs'], 
                validation_data=(test_data, test_label), 
                shuffle=True, 
                callbacks=[tensorboard, cp_callback])
    
    # save the model of the last epoch
    model.save("models/efficientnet_last_" + config.options['logs_name'] + ".h5")
    print("***Efficientnet Model Training Finished***")
    plot_hist(history_efficientnet, figure_name="efficientnet")


def train_model(train_mode):
    """This function loads train and validation data and trains the appropiate model based on trainining mode

    Args:
        train_mode (string): training mode one of resnet/efficeintnet/ensemble
    """  
    # training_mode = config.options['train_mode']
    # generate training data
    train_data, train_label = generate_train_data(config.paths['train_data'])
    train_data_len = len(train_data)
    print(f"***Total Train Data {train_data_len}***")

    # generate validation data
    test_data, test_label = process_test_data(config.paths['test_data'])
    val_data_len = len(test_data)
    print(f"***Total Validation Data {val_data_len}***")

    # monitor model logs on tensorboard
    tensorboard = TensorBoard(log_dir="logs/{}".format(config.options['logs_name']))

    if train_mode == "resnet":
        train_resnet_model(train_data, train_label, test_data, test_label, tensorboard, config.options['pre_trained'])
        
    elif train_mode == "efficientnet":
        train_efficient_net_model(train_data, train_label, test_data, test_label, tensorboard, config.options['pre_trained'])
        
    elif train_mode == "ensemble":
        train_resnet_model(train_data, train_label, test_data, test_label, tensorboard, pre_trained=False)
        train_efficient_net_model(train_data, train_label, test_data, test_label, tensorboard, pre_trained=False)
        print("***Starting the ensemble***")
        # loading the best models from trained resnet and effiecientnet model
        resnet_model = load_model('checkpoints/resnet_best_' + config.options['logs_name'] + '.h5', compile=False)
        efficientnet_model = load_model('checkpoints/efficientnet_best_' + config.options['logs_name'] + '.h5', compile=False)

        # create a list of the models
        models = [resnet_model, efficientnet_model]
        # define the input shape of the model
        model_input = tf.keras.Input(shape=(224, 224, 3))
        # get the output layer of the model
        model_outputs = [model(model_input) for model in models]
        # define average ensemble method on the two modes
        ensemble_output = tf.keras.layers.Average()(model_outputs)
        # define the ensemble model
        ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)  
        # save the ensemble model
        ensemble_model.save('checkpoints/ensemble_model_' + config.options['logs_name'] + '.h5')
        print("finished Ensembling model")


if __name__ == "__main__":
    train_model(config.options['train_mode'])