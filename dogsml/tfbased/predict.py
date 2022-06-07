import collections
import numpy as np
import os
import tensorflow as tf

import dogsml.settings
import dogsml.transfer.constants
import dogsml.utils
import dogsml.tfbased.utils


__all__ = (
    "load_model",
    "predict_folder",
    "predict_from_url",
    "interpret_predicted_dog_values",
    "interpret_categorical_results",
    "interpret_result_with_probabilities",
)


def load_model(model_name):
    """
    Load and return tf.Model
    :param model_name: (str)
    :return: (tf.Model)
    """
    tf_model = tf.keras.models.load_model(
        "{0}/{1}".format(
            dogsml.settings.COMPILED_MODELS_PATH,
            model_name,
        )
    )
    return tf_model


def predict_folder(image_folder, tf_model, img_width, img_height):
    """
    Run model on the image folder.
    Return numpy array with probabilities (0 < p < 1)
    :param image_folder: (str)
    :param tf_model: (tf.Model)
    :param img_width: (int)
    :param img_height: (int)
    :return: (ndarray)
    """
    image_data = []
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        image_data.append(
            dogsml.utils.dataset.extract_image_data_from_path(
                image_path,
                width=img_width,
                height=img_height,
            )
        )
    image_data = np.array(image_data)
    return tf_model.predict(image_data)


def predict_from_url(url, tf_model, img_width, img_height):
    """
    Run tf Model on image url
    :param url: (str)
    :param tf_model: (tf.Model)
    :param img_width: (int)
    :param img_height: (int)
    :return: (List[float])
    """
    data = dogsml.tfbased.utils.read_tensor_from_image_url(
        url,
        input_height=img_height,
        input_width=img_width,
        scale=False,
    )
    predicted_values = tf_model.predict(data)
    return predicted_values


def interpret_predicted_dog_values(value):
    """
    Convert probability to human like language
    :param value: float
    :return: str
    """
    if value >= 0.5:
        return "It is a dog"
    return "It is not a dog"


def interpret_categorical_results(results, class_names):
    """
    Useful together with predict_folder
    :param results: (ndarray) - array of one-hot vectors
    :param class_names: (list) - names of categories
    :return: (dict)
        a_class: 10
        b_class: 5
    """
    counter = collections.Counter()
    for item in results:
        value = np.argmax(item)
        counter[class_names[value]] += 1
    return counter


def interpret_result_with_probabilities(result, class_names, tops):
    """
    :param result: (ndarray) - array of one-hot vectors
    :param class_names: (list) - names of categories
    :param tops: (int) - how many top results to return
    :return: (dict)
        top 1 class: probability %
        top 2 class: probability %
        ...
    """
    probabilities = dict()
    result = result[0]
    indices = np.argpartition(result, -tops)[-tops:]
    for value in indices:
        probabilities[class_names[value]] = result[value]
    return probabilities


def predict_dog_breed(url, tf_model, img_height, img_width):
    result = predict_from_url(url, tf_model, img_height, img_width)
    probabilities = interpret_result_with_probabilities(
        result,
        dogsml.transfer.constants.DOG_BREEDS,
        3,
    )
    for label, value in probabilities.items():
        printed_value = format(value * 100, ".2f")
        print("{0} --- {1}%".format(label, printed_value))


def predict_transfer_model(
    folder,
    tf_model,
    img_width,
    img_height,
    class_names,
):
    image_data = []
    for filename in os.listdir(folder):
        if filename.startswith("."):
            continue
        image_path = os.path.join(folder, filename)
        image_data.append(
            dogsml.utils.dataset.extract_image_data_from_path(
                image_path,
                width=img_width,
                height=img_height,
                scale=False,
            )
        )
    image_data = np.array(image_data)
    results = tf_model.predict(image_data)
    counter = interpret_categorical_results(results, class_names)
    return counter
