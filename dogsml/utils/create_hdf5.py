import h5py

import dataset


def create_natural_images_file(convolutional=False):
    """
    Create hdf5 file with 6 datasets:
        x_dev , y_dev
        x_train, y_train
        x_test, y_test
    :param convolutional: (bool) Specify output data format:
        False: dataset.shape = (number of examples, number of parameters)
        True: dataset.shape = (number of examples, width, height, channels)
    :return: (None)
    """
    if convolutional:
        x_dev, y_dev = dataset.extract_image_data("dev")
        x_test, y_test = dataset.extract_image_data("test")
        x_train, y_train = dataset.extract_image_data("train")
    else:
        x_dev, y_dev = dataset.prepare_images("dev")
        x_test, y_test = dataset.prepare_images("test")
        x_train, y_train = dataset.prepare_images("train")
        x_dev = x_dev.T
        x_test = x_test.T
        x_train = x_train.T
        y_dev = y_dev.T
        y_test = y_test.T
        y_train = y_train.T

    filename = dataset.NATURAL_IMAGES_HDF5
    if convolutional:
        filename = dataset.NATURAL_IMAGES_HDF5_CONV
    with h5py.File(filename, "w") as hf:
        hf.create_dataset(
            "x_dev",
            data=x_dev,
            shape=x_dev.shape,
            compression="gzip",
            chunks=True,
        )
        hf.create_dataset(
            "y_dev",
            data=y_dev,
            shape=y_dev.shape,
            compression="gzip",
            chunks=True,
        )
        hf.create_dataset(
            "x_test",
            data=x_test,
            shape=x_test.shape,
            compression="gzip",
            chunks=True,
        )
        hf.create_dataset(
            "y_test",
            data=y_test,
            shape=y_test.shape,
            compression="gzip",
            chunks=True,
        )
        hf.create_dataset(
            "x_train",
            data=x_train,
            shape=x_train.shape,
            compression="gzip",
            chunks=True,
        )
        hf.create_dataset(
            "y_train",
            data=y_train,
            shape=y_train.shape,
            compression="gzip",
            chunks=True,
        )


if __name__ == '__main__':
    create_natural_images_file(convolutional=False)
    create_natural_images_file(convolutional=True)
