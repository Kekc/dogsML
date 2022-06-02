import h5py
import numpy as np
import os

import dogsml.settings
import dogsml.utils

from dogsml.utils.dataset import NATURAL_IMAGES_CLASS_NAMES


def create_categorical_hdf5(image_folder):
    """
    Create h5 file with categorical data
    image folder structure:
    - image folder
      - class_a
      - class_b

    Output file structure:
    x_data: shape (num examples, width, height, channels)
    y_data: one-hot representation, shape(num examples, num classes)
    :param image_folder: (str)
    :return: (None)
    """
    x_data = []
    y_data = []
    categories = [
        item for item in os.listdir(image_folder) if not item.startswith(".")
    ]
    category_count = len(categories)
    for category_name in categories:
        category_index = NATURAL_IMAGES_CLASS_NAMES.index(category_name)
        for filename in os.listdir(
                os.path.join(image_folder, category_name)
        ):
            image_path = os.path.join(image_folder, category_name, filename)
            image_data = dogsml.utils.dataset.extract_image_data_from_path(
                image_path
            )
            image_data = np.array(image_data)
            x_data.append(image_data)
            one_hot_value = np.zeros(category_count)
            one_hot_value[category_index] = 1
            y_data.append(one_hot_value)

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    h5_filename = os.path.join(
        dogsml.settings.DATA_ROOT,
        "{0}.hdf5".format(image_folder),
    )

    with h5py.File(h5_filename, "w") as hf:
        hf.create_dataset(
            "x_data",
            data=x_data,
            shape=x_data.shape,
            compression="gzip",
            chunks=True,
        )
        hf.create_dataset(
            "y_data",
            data=y_data,
            shape=y_data.shape,
            compression="gzip",
            chunks=True,
        )


if __name__ == '__main__':
    create_categorical_hdf5(
        os.path.join(dogsml.settings.DATA_ROOT, "natural_images")
    )
