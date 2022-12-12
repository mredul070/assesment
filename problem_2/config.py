paths = dict(
    train_data = "dataset_256X256/test_dir",
    test_data = "dataset_256X256/test",
    inference_img_path = "dataset_256X256/test_dir/berry/1_256.jpg",
    pre_trained_model_path = "",
    inference_model_path = "models/test_model2.h5",
)

train_params = dict(
    batch_size = 2,
    epochs = 300,
)

options = dict(
    train_mode = "resnet",     # choose one of the following three : resnet/efficientnet/ensemble
    pre_trained = False,
    logs_name = "resnet"
)