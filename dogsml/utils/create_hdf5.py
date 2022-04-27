import h5py

import dataset


def create_natural_images_file():
    """
    Create hdf5 file with 6 datasets:
        x_dev , y_dev
        x_train, y_train
        x_test, y_test
    Shape: (number of examples, number of parameters)
    :return: (None)
    """
    x_dev, y_dev = dataset.prepare_images("dev")
    x_test, y_test = dataset.prepare_images("test")
    x_train, y_train = dataset.prepare_images("train")

    with h5py.File(
        "{0}/natural_images.hdf5".format(dataset.DATASET_FOLDER),
        "w",
    ) as hf:
        hf.create_dataset(
            "x_dev",
            data=x_dev.T,
            shape=x_dev.T.shape,
            compression="gzip",
            chunks=True,
        )
        hf.create_dataset(
            "y_dev",
            data=y_dev.T,
            shape=y_dev.T.shape,
            compression="gzip",
            chunks=True,
        )
        hf.create_dataset(
            "x_test",
            data=x_test.T,
            shape=x_test.T.shape,
            compression="gzip",
            chunks=True,
        )
        hf.create_dataset(
            "y_test",
            data=y_test.T,
            shape=y_test.T.shape,
            compression="gzip",
            chunks=True,
        )
        hf.create_dataset(
            "x_train",
            data=x_train.T,
            shape=x_train.T.shape,
            compression="gzip",
            chunks=True,
        )
        hf.create_dataset(
            "y_train",
            data=y_train.T,
            shape=y_train.T.shape,
            compression="gzip",
            chunks=True,
        )


if __name__ == '__main__':
    create_natural_images_file()
