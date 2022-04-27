import numpy as np
import os
import csv
import cv2


__all__ = (
    "PROJECT_ROOT",
    "DATA_ROOT",
    "IMG_FOLDER",
    "DATASET_FOLDER",
    "prepare_dataset",
    "prepare_images",
)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
IMG_FOLDER = "{0}/natural_images".format(DATA_ROOT)
DATASET_FOLDER = "{0}/dogs_dataset".format(DATA_ROOT)


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
    for directory in os.listdir(IMG_FOLDER):
        if directory.startswith("."):
            continue
        is_dog = directory == "dog"
        for filename in os.listdir(os.path.join(IMG_FOLDER, directory)):
            image_path = os.path.join(IMG_FOLDER, directory, filename)
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

    with open("{0}/dev.csv".format(DATASET_FOLDER), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["PATH", "IS_DOG"])
        for path in dog_paths[:dogs_10pc]:
            writer.writerow([path, 1])
        for path in non_dog_paths[:non_dogs_10pc]:
            writer.writerow([path, 0])

    with open("{0}/test.csv".format(DATASET_FOLDER), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["PATH", "IS_DOG"])
        for path in dog_paths[dogs_10pc: 2 * dogs_10pc]:
            writer.writerow([path, 1])
        for path in non_dog_paths[non_dogs_10pc: 2 * non_dogs_10pc]:
            writer.writerow([path, 0])

    with open("{0}/train.csv".format(DATASET_FOLDER), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["PATH", "IS_DOG"])
        for path in dog_paths[2 * dogs_10pc:]:
            writer.writerow([path, 1])
        for path in non_dog_paths[2 * non_dogs_10pc:]:
            writer.writerow([path, 0])


def prepare_images(filename, width=64, height=64):
    """
    Parse csv file and prepare two numpy arrays:
    x - data
    y - true "label" vector
    :param filename: (str)
    :return: (x_dev_flatten, y_dev)
        x_dev_flatten : (ndarray) of shape
          (number of parameters: width * height * channels, number of examples)
        y_dev : (ndarray) of shape (1, number of examples)

    """
    x_dev = []
    y_dev = []
    with open("{0}/{1}.csv".format(DATASET_FOLDER, filename), "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for image_path, value in reader:
            image_path = os.path.join(PROJECT_ROOT, image_path)
            img_arr = cv2.imread(image_path)
            img_arr = cv2.resize(img_arr, (width, height))
            x_dev.append(img_arr)
            y_dev.append(int(value))
    x_dev = np.asarray(x_dev)

    # x shape: (num examples, width, height, channels)
    # y shape: (num examples)

    y_dev = np.array(y_dev).reshape(1, -1)
    x_dev = x_dev / 255
    x_dev_flatten = x_dev.reshape(x_dev.shape[0], -1).T

    # x shape: (parameters, num examples)
    # y shape: (1, num examples)
    return x_dev_flatten, y_dev
