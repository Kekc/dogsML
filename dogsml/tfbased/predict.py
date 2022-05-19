import numpy as np
import os
import tensorflow as tf

import dogsml.settings
import dogsml.utils
import dogsml.tfbased.utils


__all__ = (
    "predict",
    "predict_from_url",
    "interpret_predicted_dog_values",
)


def predict(image_folder):
    """
    Run model on the image folder.
    Return numpy array with probabilities (0 < p < 1)
    :param image_folder: str
    :return: ndarray
    """
    image_data = []
    image_names = []
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        image_names.append(filename)
        image_data.append(
            dogsml.utils.dataset.extract_image_data_from_path(image_path)
        )
    image_data = np.array(image_data)
    conv_model = tf.keras.models.load_model(
        "{0}/conv_model_500".format(
            dogsml.settings.COMPILED_MODELS_PATH
        )
    )
    return conv_model.predict(image_data)


def predict_from_url(url):
    """
    Load TF model and run it using image url
    :param url: (str)
    :return: float
    """
    data = dogsml.tfbased.utils.read_tensor_from_image_url(url)
    conv_model = tf.keras.models.load_model(
        "{0}/conv_model_500".format(
            dogsml.settings.COMPILED_MODELS_PATH
        )
    )
    predicted_values = conv_model.predict(data)
    return predicted_values[0][0]


def interpret_predicted_dog_values(value):
    """
    Convert probability to human like language
    :param value: float
    :return: str
    """
    if value >= 0.5:
        return "It is a dog"
    return "It is not a dog"
