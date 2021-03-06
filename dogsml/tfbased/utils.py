import requests
import tensorflow as tf


def forward_step(x_set, w1, b1, w2, b2):
    """
    Run one forward step of neural net using tensorflow
    :param x_set: (ndarray) of shape (number of parameters, number of examples)
    :param w1: (tf.Tensor) of shape (layer1_size, number of x parameters)
    :param b1: (tf.Tensor) of shape (layer1_size, 1)
    :param w2: (tf.Tensor) of shape (1, layer1_size)
    :param b2: (tf.Tensor) of shape (1, 1)
    :return: a2: (tf.Tensor) of shape (1, number of examples)
        Last layer's activations: probabilities of the predicted values
    """
    z1 = tf.add(tf.linalg.matmul(w1, x_set), b1)
    a1 = tf.keras.activations.relu(z1)
    z2 = tf.add(tf.linalg.matmul(w2, a1), b2)
    a2 = tf.keras.activations.sigmoid(z2)
    return a2


def initialize_params(x_train, y_train, layer1_size):
    """
    Initialize neural net parameters for tensorflow usage
    :param x_train: (ndarray) of shape
        (number of parameters, number of examples) Training data
    :param y_train: (ndarray) of shape
        (1, number of examples) Training true values
    :param layer1_size: (int)
    :return: (w1, b1, w2, b2)
        w1 : (tf.Tensor) of shape (layer1_size, number of x parameters)
        b1 : (tf.Tensor) of shape (layer1_size, 1)
        w2 : (tf.Tensor) of shape (1, layer1_size)
        b2 : (tf.Tensor) of shape (1, 1)
    """
    initializer = tf.keras.initializers.GlorotNormal()
    input_layer_size = x_train.shape[0]
    output_layer_size = y_train.shape[0]

    w1 = tf.Variable(initializer(shape=(layer1_size, input_layer_size)))
    b1 = tf.Variable(initializer(shape=(layer1_size, 1)))
    w2 = tf.Variable(initializer(shape=(output_layer_size, layer1_size)))
    b2 = tf.Variable(initializer(shape=(output_layer_size, 1)))
    return w1, b1, w2, b2


def read_tensor_from_image_url(
    url,
    input_height=64,
    input_width=64,
    input_std=255,
    scale=True,
):
    """
    Download image from url and prepare a tf.Tensor
    :param url: (str)
    :param input_height: (int)
    :param input_width: (height)
    :param input_std: (int) - max value of the single channel
    :param scale: (bool) - Set True to normalize values to interval [0; 1]
    :return: (tf.Tensor: shape=(1, 64, 64, 3), dtype=float32)
    """
    image_reader = tf.image.decode_jpeg(
        requests.get(url).content, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize(dims_expander, [input_height, input_width])
    if scale:
        return tf.divide(resized, [input_std])
    return resized
