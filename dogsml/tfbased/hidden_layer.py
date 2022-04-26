import tensorflow as tf

import dogsml.utils
import utils


def neural_net_model(x_train, y_train, x_test, y_test, layer1_size=1, num_epochs=100, learning_rate=0.0001):

    examples_count = y_train.shape[1]
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    train_accuracy = tf.keras.metrics.BinaryAccuracy()
    test_accuracy = tf.keras.metrics.BinaryAccuracy()

    w1, b1, w2, b2 = utils.initialize_params(x_train, y_train, layer1_size)

    for epoch in range(num_epochs):
        train_accuracy.reset_states()

        with tf.GradientTape() as tape:
            a2 = utils.forward_step(x_train, w1, b1, w2, b2)
            loss = tf.keras.losses.binary_crossentropy(y_train, a2)
            cost = tf.reduce_mean(loss)

        train_accuracy.update_state(y_train, a2)
        trainable_variables = [w1, b1, w2, b2]
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))

        epoch_cost = cost / examples_count

        if not epoch % 50:
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            print("Train accuracy:", train_accuracy.result())

            test_a2 = utils.forward_step(x_test, w1, b1, w2, b2)
            test_accuracy.update_state(y_test, test_a2)
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
        layer1_size=4,
        num_epochs=401,
        learning_rate=0.09
    )
