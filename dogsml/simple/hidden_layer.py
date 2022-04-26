import numpy as np

import dogsml.utils


def propagate_forward(x_set, params):

    z1 = np.dot(params["w1"], x_set) + params["b1"]
    # relu
    a1 = dogsml.utils.functions.relu(z1)
    # tanh
    # a1 = np.tanh(z1)
    z2 = np.dot(params["w2"], a1) + params["b2"]
    a2 = dogsml.utils.functions.sigmoid(z2)

    return {
        "a1": a1,
        "a2": a2
    }


def propagate_backward(x_set, y_set, cache, params):
    examples_count = y_set.shape[1]

    dz2 = cache["a2"] - y_set
    dw2 = np.dot(dz2, cache["a1"].T) / examples_count
    db2 = np.sum(dz2, axis=1, keepdims=True) / examples_count

    # relu
    da1 = np.dot(params["w2"].T, dz2)
    dz1 = np.multiply(da1, np.int64(cache["a1"] > 0))
    dw1 = np.dot(dz1, x_set.T) / examples_count
    db1 = np.sum(dz1, axis=1, keepdims=True) / examples_count

    # tanh
    # dz1 = np.dot(params["w2"].T, dz2) * (1 - np.power(cache["a1"], 2))
    # dw1 = np.dot(dz1, x_set.T) / examples_count
    # db1 = np.sum(dz1, axis=1, keepdims=True) / examples_count

    return {
        "dw1": dw1,
        "db1": db1,
        "dw2": dw2,
        "db2": db2,
    }


def update_params(params, grads, learning_rate=0.1):
    w1 = params["w1"] - learning_rate * grads["dw1"]
    w2 = params["w2"] - learning_rate * grads["dw2"]
    b1 = params["b1"] - learning_rate * grads["db1"]
    b2 = params["b2"] - learning_rate * grads["db2"]

    return {
        "w1": w1,
        "w2": w2,
        "b1": b1,
        "b2": b2,
    }


def neural_net_model(x_set, y_set, hidden_layer_size=4, num_iterations=600, learning_rate=0.1):
    """

    :param hidden_layer_size:
    :param x_dev:
    :param y_dev:
    :param num_iterations:
    :param learning_rate:
    :return:
    """
    input_layer_size = x_set.shape[0]
    output_layer_size = y_set.shape[0]
    examples_count = y_set.shape[1]
    accuracy_set = []

    w1 = np.random.randn(hidden_layer_size, input_layer_size) * 0.01
    b1 = np.zeros((hidden_layer_size, 1))
    w2 = np.random.randn(output_layer_size, hidden_layer_size) * 0.01
    b2 = np.zeros((output_layer_size, 1))
    params = {
        "w1": w1,
        "w2": w2,
        "b1": b1,
        "b2": b2,
    }

    for iteration in range(num_iterations):
        cache = propagate_forward(x_set, params)
        cost = dogsml.utils.functions.get_cost(cache["a2"], y_set, examples_count)
        cost = float(np.squeeze(cost))
        if cost == 0:
            break
        grads = propagate_backward(x_set, y_set, cache, params)
        params = update_params(params, grads, learning_rate)

        if not iteration % 1:
            y_pred = np.where(cache["a2"] < 0.5, 0, 1)
            accuracy = 100 - np.mean(np.abs(y_pred - y_set)) * 100
            accuracy_set.append(accuracy)
            print("iteration", iteration)
            print("cost", cost)
            print("accuracy", accuracy)
    return params


if __name__ == '__main__':
    x_dev, y_dev = dogsml.utils.dataset.prepare_images("dev")
    x_test, y_test = dogsml.utils.dataset.prepare_images("test")
    params = neural_net_model(x_dev, y_dev, hidden_layer_size=2, num_iterations=100, learning_rate=0.2)
    cache = propagate_forward(x_test, params)
    predictions = np.where(cache["a2"] < 0.5, 0, 1)
    accuracy_test = 100 - np.mean(np.abs(predictions - y_test)) * 100
    print("Accuracy on the test set: ", accuracy_test)
