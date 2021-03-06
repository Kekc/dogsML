import numpy as np
import os
import csv
import cv2

import dogsml.settings


__all__ = (
    "NATURAL_IMAGES_CLASS_NAMES",
    "extract_image_data",
    "extract_image_data_from_path",
    "prepare_dataset",
    "prepare_images",
)

NATURAL_IMAGES_CLASS_NAMES = [
    "airplane",
    "car",
    "cat",
    "dog",
    "flower",
    "fruit",
    "motorbike",
    "person",
]


def prepare_dataset():
    """
    Collect paths from different image directories.
    Split values into:
     - Training set (80%)
     - Dev set (10%)
     - Test set (10%)
    Create csv files in the following format:
    `FILE_PATH, VALUE`
    :return: None
    """
    dog_paths = []
    non_dog_paths = []
    all_images_count = 0
    dogs_count = 0
    non_dogs_count = 0
    for directory in os.listdir(dogsml.settings.IMG_FOLDER):
        if directory.startswith("."):
            continue
        is_dog = directory == "dog"
        for filename in os.listdir(
            os.path.join(dogsml.settings.IMG_FOLDER, directory)
        ):
            image_path = os.path.join(
                dogsml.settings.IMG_FOLDER,
                directory,
                filename
            )
            if is_dog:
                dog_paths.append(image_path)
                dogs_count += 1
            else:
                non_dog_paths.append(image_path)
                non_dogs_count += 1
            all_images_count += 1

    dogs_10pc = dogs_count // 10
    non_dogs_10pc = non_dogs_count // 10

    np.random.shuffle(dog_paths)
    np.random.shuffle(non_dog_paths)

    with open("{0}/dev.csv".format(
        dogsml.settings.DATASET_FOLDER), "w"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["PATH", "IS_DOG"])
        for path in dog_paths[:dogs_10pc]:
            writer.writerow([path, 1])
        for path in non_dog_paths[:non_dogs_10pc]:
            writer.writerow([path, 0])

    with open("{0}/test.csv".format(dogsml.settings.DATASET_FOLDER), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["PATH", "IS_DOG"])
        for path in dog_paths[dogs_10pc: 2 * dogs_10pc]:
            writer.writerow([path, 1])
        for path in non_dog_paths[non_dogs_10pc: 2 * non_dogs_10pc]:
            writer.writerow([path, 0])

    with open("{0}/train.csv".format(
        dogsml.settings.DATASET_FOLDER), "w"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["PATH", "IS_DOG"])
        for path in dog_paths[2 * dogs_10pc:]:
            writer.writerow([path, 1])
        for path in non_dog_paths[2 * non_dogs_10pc:]:
            writer.writerow([path, 0])


def extract_image_data(filename, width=64, height=64):
    """
    Parse csv file and prepare two numpy arrays:
    x - data
    y - true "label" vector
    :param filename: (str)
    :param width: (int)
    :param height: (int)
    :return: (x_dev, y_dev)
        x_dev : (ndarray) (num examples, width, height, channels)
        y_dev : (ndarray) (num examples, 1)
    """
    x_dev = []
    y_dev = []
    with open("{0}/{1}.csv".format(
        dogsml.settings.DATASET_FOLDER, filename), "r"
    ) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for image_path, value in reader:
            image_path = os.path.join(dogsml.settings.PROJECT_ROOT, image_path)
            img_arr = cv2.imread(image_path)
            img_arr = cv2.resize(img_arr, (width, height))
            x_dev.append(img_arr)
            y_dev.append(int(value))
    x_dev = np.asarray(x_dev)
    x_dev = x_dev / 255
    y_dev = np.array(y_dev).reshape(-1, 1)
    return x_dev, y_dev


def extract_image_data_from_path(image_path, width=64, height=64, scale=True):
    """
    Return numpy array with image data
    :param image_path: (str)
    :param width: (int)
    :param height: (int)
    :param scale: (bool) set True to scale pixel values to [0;1]
    :return: (ndarray)
    """
    img_arr = cv2.imread(image_path)
    img_arr = cv2.resize(img_arr, (width, height))
    img_arr = np.asarray(img_arr)
    if scale:
        img_arr = img_arr / 255
    return img_arr


def prepare_images(filename, width=64, height=64):
    """
    Flatten and reshape image data
    :param filename: (str)
    :param width: (int)
    :param height: (int)
    :return: (x_dev_flatten, y_dev)
        x_dev_flatten : (ndarray) of shape
          (number of parameters: width * height * channels, number of examples)
        y_dev : (ndarray) of shape (1, number of examples)

    """
    x_dev, y_dev = extract_image_data(filename, width, height)
    # x shape: (num examples, width, height, channels)
    # y shape: (num examples, 1)

    x_dev_flatten = x_dev.reshape(x_dev.shape[0], -1).T

    # x shape: (parameters, num examples)
    # y shape: (1, num examples)
    return x_dev_flatten, y_dev.T
