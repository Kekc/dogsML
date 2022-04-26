import tensorflow as tf


def forward_step(x_set, w1, b1, w2, b2):
    z1 = tf.add(tf.linalg.matmul(w1, x_set), b1)
    a1 = tf.keras.activations.relu(z1)
    z2 = tf.add(tf.linalg.matmul(w2, a1), b2)
    a2 = tf.keras.activations.sigmoid(z2)
    return a2


def initialize_params(x_train, y_train, layer1_size):
    initializer = tf.keras.initializers.GlorotNormal()
    input_layer_size = x_train.shape[0]
    output_layer_size = y_train.shape[0]

    w1 = tf.Variable(initializer(shape=(layer1_size, input_layer_size)))
    b1 = tf.Variable(initializer(shape=(layer1_size, 1)))
    w2 = tf.Variable(initializer(shape=(output_layer_size, layer1_size)))
    b2 = tf.Variable(initializer(shape=(output_layer_size, 1)))
    return w1, b1, w2, b2
