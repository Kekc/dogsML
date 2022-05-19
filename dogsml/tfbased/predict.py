import numpy as np
import os
import tensorflow as tf

import dogsml.utils


def predict(image_folder):
    image_data = []
    image_names = []
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        image_names.append(filename)
        image_data.append(
            dogsml.utils.dataset.extract_image_data_from_path(image_path)
        )
    image_data = np.array(image_data)
    print("image_data.shape", image_data.shape)
    conv_model = tf.keras.models.load_model(
        "{0}/conv_model_500".format(
            dogsml.utils.dataset.COMPILED_MODELS_PATH
        )
    )
    predicted_values = conv_model.predict(image_data)
    for index in range(len(image_names)):
        print("{0} === {1}".format(
            image_names[index],
            predicted_values[index]),
        )
    return conv_model.predict(image_data)


if __name__ == '__main__':
    predict("{0}/dog_breeds/images/Images/n02085620-Chihuahua".format(
        dogsml.utils.dataset.DATA_ROOT)
    )
