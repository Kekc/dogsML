import h5py
import tensorflow as tf
import tensorflow.keras.layers as tfl

import dogsml.utils


def neural_net_model(
        x_train,
        y_train,
        x_test,
        y_test,
        num_epochs=100,
        batch_size=64,
):
    """
    Implement convolutional neural net using tensorflow API
    :param x_train: (ndarray) of shape
        (number of parameters, number of examples) Training data
    :param y_train: (ndarray) of shape
        (1, number of examples) Training true values
    :param x_test: (ndarray) of shape
        (number of parameters, number of examples) Test true values
    :param y_test: (ndarray) of shape (1, number of examples) Test data
    :param num_epochs: (int)
    :param batch_size: (int)
    :return:
    """
    input_layer = tf.keras.Input(shape=x_train.shape[1:])

    z1 = tfl.Conv2D(8, 4, padding="same")(input_layer)
    a1 = tfl.ReLU()(z1)
    p1 = tfl.MaxPool2D(pool_size=(8, 8), strides=(8, 8), padding='same')(a1)
    z2 = tfl.Conv2D(16, 2, padding="same")(p1)
    a2 = tfl.ReLU()(z2)
    p2 = tfl.MaxPool2D(pool_size=(4, 4), strides=(4, 4), padding='same')(a2)
    f1 = tfl.Flatten()(p2)
    output_layer = tfl.Dense(1, activation="sigmoid")(f1)

    conv_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    conv_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)
    ).batch(batch_size)

    conv_model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=test_dataset
    )


def run_model():
    """
    Service method to run neural net
    :return: (None)
    """
    dataset = h5py.File(dogsml.utils.dataset.NATURAL_IMAGES_HDF5_CONV, "r")
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    x_test = dataset["x_test"]
    y_test = dataset["y_test"]

    neural_net_model(
        x_train,
        y_train,
        x_test,
        y_test,
        num_epochs=500,
        batch_size=32,
    )


if __name__ == '__main__':
    run_model()
