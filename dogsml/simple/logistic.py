import numpy as np
import matplotlib.pyplot as plt

import dogsml.utils


def logistic_regression(
        x_dev,
        y_dev,
        num_iterations=100,
        learning_rate=0.009,
        accuracy_plot=False
):
    """
    Run logistic regression and find optimal values for `w` and `b`

    :param x_dev: (ndarray) of shape (number of parameters, number of examples)
    :param y_dev: (ndarray) of shape (1, number of examples)
    :param num_iterations: (int)
    :param learning_rate: (float)
    :param accuracy_plot: (bool)
    :return: (w, b)
        w : (ndarray) of shape (number of x parameters, 1)
        b : (float)
    """

    accuracy_set = []
    w = np.zeros((x_dev.shape[0], 1))
    b = 0.0

    for iteration in range(num_iterations):
        grads, cost, accuracy = propagate_forward(x_dev, y_dev, w, b)
        if not iteration % 100:
            accuracy_set.append(accuracy)
            print("iteration", iteration)
            print("accuracy", accuracy)
        if cost == 0:
            break
        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]

    if accuracy_plot:
        plt.plot(accuracy_set)
        plt.show()

    return w, b


def propagate_forward(data, true_values, w, b):
    """
    Run one step of logistic regression,
    count derivatives for parameters `w` and `b`
    Count cost and accuracy for the current step

    :param data: (ndarray) of shape (number of parameters, number of examples)
    :param true_values: (ndarray) of shape (1, number of examples)
    :param w: (ndarray) of shape (number of x parameters, 1)
    :param b: (float)
    :return: (grads, cost, accuracy)
        grads : (dict(dw=dw, db=db))
            dw : (ndarray) of shape (number of x parameters, 1)
            db : (float)
        cost : (ndarray) of shape ()
        accuracy : (float)
    """
    examples_count = data.shape[1]
    activation = dogsml.utils.functions.activation_function(data, w, b)

    if np.equal(activation, true_values).all():
        cost = 0
    else:
        cost = dogsml.utils.functions.get_cost(
            activation,
            true_values,
            examples_count
        )

    dw = (1 / examples_count) * np.dot(data, (activation - true_values).T)
    db = (1 / examples_count) * np.sum(activation - true_values)

    cost = np.squeeze(np.array(cost))

    grads = {
        "dw": dw,
        "db": db,
    }

    accuracy = dogsml.utils.functions.get_accuracy(data, true_values, w, b)

    return grads, cost, accuracy


if __name__ == '__main__':
    x_dev, y_dev = dogsml.utils.dataset.prepare_images("dev")
    x_test, y_test = dogsml.utils.dataset.prepare_images("test")
    w, b = logistic_regression(
        x_dev,
        y_dev,
        num_iterations=600,
        learning_rate=0.1,
    )
    accuracy_test = dogsml.utils.functions.get_accuracy(x_test, y_test, w, b)
    print("Accuracy on the test set: ", accuracy_test)
