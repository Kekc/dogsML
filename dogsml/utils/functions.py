import numpy as np
import tensorflow as tf


def sigmoid(z):
    """
    :param z: (ndarray)
    :return: (ndarray)
    """
    return 1 / (1 + np.exp(-z))


def relu(x):
    """
    Rectified Linear Unit activation function
    :param x: (ndarray)
    :return: (ndarray)
    """
    return x * (x > 0)


def shuffle_array(x_array):
    """
    Randomly shuffle numpy array

    :param x_array: (ndarray)
    :return: (ndarray)
    """
    shuffler = np.random.permutation(len(x_array))
    return x_array[shuffler]


def get_cost(activation, true_values, examples_count):
    """
    :param activation: (ndarray) of shape (1, number of examples)
    :param true_values: (ndarray) of shape (1, number of examples)
    :param examples_count: (int)
    :return:
    """
    logprobs = true_values * np.log(activation) + (1 - true_values) * np.log(1 - activation)
    return -np.sum(logprobs) / examples_count


def activation_function(data, w, b):
    """
    Normalize values in range (0, 1)
    :param data: (ndarray) of shape (number of parameters, number of examples)
    :param w: (ndarray) of shape (number of x parameters, 1)
    :param b: (float)
    :return: (ndarray) of shape (1, number of examples)
    """
    return sigmoid(np.dot(w.T, data) + b)


def get_accuracy(x_test, y_test, w, b):
    """
    Get accuracy for given parameters `w` and `b`
    :param x_test: (ndarray) of shape (number of parameters, number of examples)
    :param y_test: (ndarray) of shape (1, number of examples)
    :param w: (ndarray) of shape (number of x parameters, 1)
    :param b: (float)
    :return: (float)
    """
    y_pred = np.where(activation_function(x_test, w, b) > 0.5, 1, 0)
    accuracy_test = 100 - np.mean(np.abs(y_pred - y_test)) * 100
    return accuracy_test


def one_hot_matrix(label, depth=6):
    """
    Computes the one hot encoding for a single label
    :param label: (int)
    :param depth: (int) Number of different classes that label can take
    :return: (tf.Tensor) A single-column matrix with the one hot encoding
    """
    t1 = tf.one_hot(label, depth, axis=0)
    return tf.reshape(t1, (depth,))
