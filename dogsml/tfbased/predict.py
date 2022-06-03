import collections
import numpy as np
import os
import tensorflow as tf

import dogsml.settings
import dogsml.utils
import dogsml.tfbased.utils


__all__ = (
    "load_model",
    "predict_folder",
    "predict_from_url",
    "interpret_predicted_dog_values",
    "interpret_categorical_results",
)


def load_model(model_name):
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
    :param model_name: (str)
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
    Load TF model and run it using image url
    :param url: (str)
    :return: (float)
    """
    data = dogsml.tfbased.utils.read_tensor_from_image_url(
        url,
        input_height=img_height,
        input_width=img_width,
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


if __name__ == "__main__":
    class_names = dogsml.utils.dataset.NATURAL_IMAGES_CLASS_NAMES
    tf_model = load_model("transfer_h5_3")
    natural_images_dir = os.path.join(
        dogsml.settings.DATA_ROOT, "natural_images"
    )
    counter = predict_transfer_model(
        os.path.join(dogsml.settings.DATA_ROOT, "natural_images/person"),
        tf_model,
        160,
        160,
        class_names,
    )
    print(counter)
