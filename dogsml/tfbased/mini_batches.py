import tensorflow as tf

import dogsml.utils
import utils


def neural_net_model(
        x_train,
        y_train,
        x_test,
        y_test,
        layer1_size=1,
        num_epochs=100,
        learning_rate=0.0001,
        minibatch_size=32,
):
    """
    Run neural net and find optimal values for parameters `w` and `b`
    :param x_train: (ndarray) of shape
        (number of parameters, number of examples) Training data
    :param y_train: (ndarray) of shape
        (1, number of examples) Training true values
    :param x_test: (ndarray) of shape
        (number of parameters, number of examples) Test true values
    :param y_test: (ndarray) of shape (1, number of examples) Test data
    :param layer1_size: (int) Size of the neural net layer
    :param num_epochs: (int)
    :param learning_rate: (float)
    :param minibatch_size: (int)
    :return: (w1, b1, w2, b2)
        w1 : (tf.Tensor) of shape (layer1_size, number of x parameters)
        b1 : (tf.Tensor) of shape (layer1_size, 1)
        w2 : (tf.Tensor) of shape (1, layer1_size)
        b2 : (tf.Tensor) of shape (1, 1)
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    train_accuracy = tf.keras.metrics.BinaryAccuracy()
    test_accuracy = tf.keras.metrics.BinaryAccuracy()

    w1, b1, w2, b2 = utils.initialize_params(x_train, y_train, layer1_size)

    x_train = tf.data.Dataset.from_tensor_slices(
        tf.transpose(x_train.astype("float32"))
    )
    y_train = tf.data.Dataset.from_tensor_slices(
        tf.transpose(y_train.astype("float32"))
    )

    x_test = tf.data.Dataset.from_tensor_slices(
        tf.transpose(x_test.astype("float32"))
    )
    y_test = tf.data.Dataset.from_tensor_slices(
        tf.transpose(y_test.astype("float32"))
    )

    dataset = tf.data.Dataset.zip((x_train, y_train))
    test_dataset = tf.data.Dataset.zip((x_test, y_test))

    examples_count = dataset.cardinality().numpy()

    minibatches = dataset.batch(minibatch_size).prefetch(8)
    test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)

    for epoch in range(num_epochs):
        epoch_cost = 0
        train_accuracy.reset_states()

        for (minibatch_X, minibatch_Y) in minibatches:
            with tf.GradientTape() as tape:
                a2 = utils.forward_step(
                    tf.transpose(minibatch_X),
                    w1,
                    b1,
                    w2,
                    b2,
                )
                loss = tf.keras.losses.binary_crossentropy(
                    tf.transpose(minibatch_Y),
                    a2
                )
                minibatch_cost = tf.reduce_mean(loss)

            train_accuracy.update_state(minibatch_Y, a2)
            trainable_variables = [w1, b1, w2, b2]
            grads = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_cost += minibatch_cost

        epoch_cost = epoch_cost / examples_count

        if not epoch % 5:
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            print("Train accuracy:", train_accuracy.result())

            for (minibatch_X, minibatch_Y) in test_minibatches:
                a2 = utils.forward_step(
                    tf.transpose(minibatch_X),
                    w1,
                    b1,
                    w2,
                    b2,
                )
                test_accuracy.update_state(minibatch_Y, a2)
            print("Test_accuracy:", test_accuracy.result())
            test_accuracy.reset_states()

    return w1, b1, w2, b2


if __name__ == '__main__':
    x_dev, y_dev = dogsml.utils.dataset.prepare_images("dev")
    x_test, y_test = dogsml.utils.dataset.prepare_images("test")
    w1, b1, w2, b2 = neural_net_model(
        x_dev,
        y_dev,
        x_test,
        y_test,
        layer1_size=5,
        num_epochs=401,
        learning_rate=0.02,
        minibatch_size=64,
    )
