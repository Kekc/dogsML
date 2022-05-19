import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

import dogsml.settings
import dogsml.utils


def neural_net_model(input_shape):
    """
    Implement convolutional neural net using tensorflow API
    :param input_shape: (tuple) Shape of the input data
        (widht, height, channels)
    :return: (tf.model)
    """

    input_layer = tf.keras.Input(shape=input_shape)

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
        metrics=[
            "binary_accuracy",
            tf.keras.metrics.FalseNegatives(),
            tf.keras.metrics.FalsePositives(),
        ],
    )

    return conv_model


def visualize(history, num_epochs, train_examples, test_examples):
    """
    Visualize training process of the neural net
    :param history: (tf.History) Result of tf.Model.fit()
    :param num_epochs: (int)
    :param train_examples: (int)
    :param test_examples: (int)
    :return: (None)
    """
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], label="Train Loss")
    ax.plot(history.history["val_loss"], label="Test Loss")
    ax.legend(loc="upper right")
    plt.ylabel("Loss value")
    plt.xlabel("Epochs")
    plt.axis([0, num_epochs, 0, 1])
    plt.show()

    train_accuracy = np.array(history.history["binary_accuracy"]) * 100
    test_accuracy = np.array(history.history["val_binary_accuracy"]) * 100
    fig, ax = plt.subplots()
    ax.plot(
        train_accuracy,
        label="Train Accuracy %",
    )
    ax.plot(
        test_accuracy,
        label="Test Accuracy %",
    )
    ax.legend(loc="lower right")
    plt.ylabel("Accuracy value %")
    plt.xlabel("Epochs")
    plt.axis([0, num_epochs, 0, 100])
    plt.show()

    false_negatives_train = np.array(history.history["false_negatives"])
    false_negatives_train = 100 * false_negatives_train / train_examples

    false_negatives_test = np.array(history.history["val_false_negatives"])
    false_negatives_test = 100 * false_negatives_test / test_examples
    fig, ax = plt.subplots()
    ax.plot(
        false_negatives_train,
        label="Train False Negatives %",
    )
    ax.plot(
        false_negatives_test,
        label="Test False Negatives %",
    )
    ax.legend(loc="upper right")
    plt.ylabel("False Negatives %")
    plt.xlabel("Epochs")
    plt.axis([0, num_epochs, 0, 100])
    plt.show()

    false_positives_train = np.array(history.history["false_positives"])
    false_positives_train = 100 * false_positives_train / train_examples

    false_positives_test = np.array(history.history["val_false_positives"])
    false_positives_test = 100 * false_positives_test / test_examples
    fig, ax = plt.subplots()
    ax.plot(
        false_positives_train,
        label="Train False Positives %",
    )
    ax.plot(
        false_positives_test,
        label="Test False Positives %",
    )
    ax.legend(loc="upper right")
    plt.ylabel("False Positives %")
    plt.xlabel("Epochs")
    plt.axis([0, num_epochs, 0, 100])
    plt.show()


def run_model(num_epochs=100, batch_size=32):
    """
    Service method to run neural net
    :param num_epochs: (int)
    :param batch_size: (int)
    :return: (None)
    """
    dataset = h5py.File(dogsml.settings.NATURAL_IMAGES_HDF5_CONV, "r")
    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    x_test = dataset["x_test"]
    y_test = dataset["y_test"]

    conv_model = neural_net_model(x_train.shape[1:])

    train_examples = x_train.shape[0]
    test_examples = x_test.shape[0]

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)
    ).batch(batch_size)

    history = conv_model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=test_dataset,
    )

    conv_model.save("{0}/conv_model_{1}".format(
        dogsml.settings.COMPILED_MODELS_PATH,
        num_epochs)
    )

    visualize(history, num_epochs, train_examples, test_examples)


if __name__ == '__main__':
    run_model(num_epochs=500, batch_size=32)
