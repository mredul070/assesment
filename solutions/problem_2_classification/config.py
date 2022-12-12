paths = dict(
    train_data = "dataset/train",   # the base path of the training images
    test_data = "dataset/test",     # the base path of the validation images
    inference_img_path = "dataset/test/berry/10_256.jpg",    # the image on which the prediction will be done
    pre_trained_model_path = "checkpoints/efficientnet_best_iter1.h5", # path of the pre-trained model if used
    inference_model_path = "checkpoints/efficientnet_best_iter1.h5",   # the path where model inference will ke kept
)

train_params = dict(
    batch_size = 1, # batch size fot model
    epochs = 10,   # number of epoch the model will run
)

options = dict(
    train_mode = "efficientnet",     # choose one of the following three : resnet/efficientnet/ensemble
    pre_trained = False,    # defined whether to use a pre-trained model or not
    logs_name = "iter3"     # the name under which logs will be kept
)