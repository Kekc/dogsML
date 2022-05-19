import h5py
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

import dogsml.settings
import dogsml.utils


def neural_net_model(
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        num_epochs=100,
        batch_size=32,
):
    """
    Implement neural net using tensorflow Sequential model
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

    tf_model = tf.keras.Sequential([
        tfl.InputLayer(input_shape=(x_train.shape[1])),
        tfl.Dense(8, "relu"),
        tfl.Dense(4, "relu"),
        tfl.Dense(2, "relu"),
        tfl.Dense(1, "sigmoid"),
    ])

    tf_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["binary_accuracy"],
    )

    tf_model.fit(
        np.array(x_train),
        np.array(y_train),
        epochs=num_epochs,
        batch_size=batch_size
    )

    if x_test and y_test:
        print("\n\nTest accuracy:")
        tf_model.evaluate(x_test, y_test)


def run_model():
    """
    Service method to run neural net
    :return: (None)
    """
    dataset = h5py.File(
        "{0}/natural_images.hdf5".format(dogsml.settings.DATASET_FOLDER),
        "r",
    )
    x_dev = dataset["x_dev"]
    y_dev = dataset["y_dev"]
    x_test = dataset["x_test"]
    y_test = dataset["y_test"]

    neural_net_model(
        x_dev,
        y_dev,
        x_test,
        y_test,
        num_epochs=100,
        batch_size=32,
    )


if __name__ == '__main__':
    run_model()
