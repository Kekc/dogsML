import tensorflow as tf
import tensorflow.keras.layers as tfl

import dogsml.utils


def neural_net_model(
        x_train,
        y_train,
        x_test,
        y_test,
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
        tfl.InputLayer(input_shape=(x_train.shape[0])),
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
    print("Model summary", tf_model.summary())

    tf_model.fit(
        x_train.T,
        y_train.T,
        epochs=num_epochs,
        batch_size=batch_size
    )

    print("\n\nTest accuracy:")
    tf_model.evaluate(x_test.T, y_test.T)


if __name__ == '__main__':
    x_dev, y_dev = dogsml.utils.dataset.prepare_images("train")
    x_test, y_test = dogsml.utils.dataset.prepare_images("test")
    neural_net_model(
        x_dev,
        y_dev,
        x_test,
        y_test,
        num_epochs=100,
        batch_size=32,
    )
