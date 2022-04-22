import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def shuffle_array(x_array):
    """
    Randomly shuffle numpy array

    :param x_array: ndarray
    :return: ndarray
    """
    shuffler = np.random.permutation(len(x_array))
    return x_array[shuffler]