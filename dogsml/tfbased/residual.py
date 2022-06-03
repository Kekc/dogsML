import h5py
import numpy as np
import os
import pathlib
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.initializers import glorot_uniform, random_uniform

import dogsml.settings
import dogsml.utils


def convolutional_block(
    input_data,
    middle_window_shape,
    filters,
    s=2,
    training=True,
    initializer=glorot_uniform,
):
    """
    Implementation of the convolutional block
    :param input_data: (tf.Tensor) shape (examples, width, height, channels)
    :param middle_window_shape: (int)
    :param filters: (list(int)) number of filter in the conv layers
    :param s: (int) Stride
    :param training: (Bool) run in training or inference mode
    :param initializer: (tf.Initializer)
    :return: (tf.Tensor) - output of the convolutional block
        shape: (width, height, channels)
    """
    filter1, filter2, filter3 = filters

    input_shortcut = input_data

    # Main path

    input_data = tfl.Conv2D(
        filters=filter1, kernel_size=1, strides=(s, s),
        padding="valid", kernel_initializer=initializer(seed=0)
    )(input_data)
    input_data = tfl.BatchNormalization(axis=3)(input_data, training=training)
    input_data = tfl.Activation("relu")(input_data)

    # Second component of the main path
    input_data = tfl.Conv2D(
        filters=filter2, kernel_size=middle_window_shape, strides=(1, 1),
        padding="same", kernel_initializer=initializer(seed=0)
    )(input_data)
    input_data = tfl.BatchNormalization(axis=3)(input_data, training=training)
    input_data = tfl.Activation("relu")(input_data)

    # Third component of the main path
    input_data = tfl.Conv2D(
        filters=filter3, kernel_size=1, strides=(1, 1), padding="valid",
        kernel_initializer=initializer(seed=0)
    )(input_data)
    input_data = tfl.BatchNormalization(axis=3)(input_data, training=training)

    # Shortcut path
    input_shortcut = tfl.Conv2D(
        filters=filter3, kernel_size=1, strides=(s, s), padding="valid",
        kernel_initializer=initializer(seed=0)
    )(input_shortcut)
    input_shortcut = tfl.BatchNormalization(axis=3)(
        input_shortcut, training=training
    )

    # Add shortcut value to the main path
    output_data = tfl.Add()([input_data, input_shortcut])
    output_data = tfl.Activation("relu")(output_data)

    return output_data


def identity_block(
    input_data,
    middle_window_shape,
    filters,
    training=True,
    initializer=random_uniform
):
    """
    Implementation of the identity block
    :param input_data: (tf.Tensor) shape (examples, width, height, channels)
    :param middle_window_shape: (int)
    :param filters: (list(int)) number of filter in the conv layers
    :param training: (Bool) run in training or inference mode
    :param initializer: (tf.Initializer)
    :return: (tf.Tensor) - output of the identity block
        shape: (examples, width, height, channels)
    """

    filter1, filter2, filter3 = filters
    input_shortcut = input_data

    # First component of the main path
    input_data = tfl.Conv2D(
        filters=filter1, kernel_size=1, strides=(1, 1), padding="valid",
        kernel_initializer=initializer(seed=0)
    )(input_data)
    input_data = tfl.BatchNormalization(axis=3)(input_data, training=training)
    input_data = tfl.Activation("relu")(input_data)

    # Second component of the main path
    input_data = tfl.Conv2D(
        filters=filter2, kernel_size=middle_window_shape, strides=(1, 1),
        padding="same", kernel_initializer=initializer(seed=0)
    )(input_data)
    input_data = tfl.BatchNormalization(axis=3)(input_data, training=training)
    input_data = tfl.Activation("relu")(input_data)

    # Third component of the main path
    input_data = tfl.Conv2D(
        filters=filter3, kernel_size=1, strides=(1, 1), padding="valid",
        kernel_initializer=initializer(seed=0)
    )(input_data)
    input_data = tfl.BatchNormalization(axis=3)(input_data, training=training)

    # Add shortcut value to the main path
    # pass it through a RELU activation
    output_data = tfl.Add()([input_data, input_shortcut])
    output_data = tfl.Activation("relu")(output_data)

    return output_data


def residual_network(input_shape, classes):
    """
    Implement residual neural net using tensorflow API
    :param input_shape: (tuple) Shape of the input data
        (width, height, channels)
    :param classes: (int) - Number of classes
    :return: (tf.model)
    """
    input_layer = tf.keras.Input(shape=input_shape)
    input_x = tfl.Rescaling(1. / 255)(input_layer)
    input_x = tfl.ZeroPadding2D((3, 3))(input_x)

    # Stage 1
    input_x = tfl.Conv2D(
        64,
        (7, 7),
        strides=(2, 2),
        kernel_initializer=glorot_uniform(seed=0)
    )(input_x)
    input_x = tfl.BatchNormalization(axis=3)(input_x)
    input_x = tfl.Activation("relu")(input_x)
    input_x = tfl.MaxPooling2D((3, 3), strides=(2, 2))(input_x)

    # Stage 2
    input_x = convolutional_block(
        input_x,
        middle_window_shape=3,
        filters=[64, 64, 256],
        s=1,
    )
    input_x = identity_block(input_x, 3, [64, 64, 256])
    input_x = identity_block(input_x, 3, [64, 64, 256])

    # Stage 3
    input_x = convolutional_block(
        input_x,
        middle_window_shape=3,
        filters=[128, 128, 512],
        s=2,
    )
    input_x = identity_block(input_x, 3, [128, 128, 512])
    input_x = identity_block(input_x, 3, [128, 128, 512])
    input_x = identity_block(input_x, 3, [128, 128, 512])

    # Stage 4
    input_x = convolutional_block(
        input_x,
        middle_window_shape=3,
        filters=[256, 256, 1024],
        s=2,
    )
    input_x = identity_block(input_x, 3, [256, 256, 1024])
    input_x = identity_block(input_x, 3, [256, 256, 1024])
    input_x = identity_block(input_x, 3, [256, 256, 1024])
    input_x = identity_block(input_x, 3, [256, 256, 1024])
    input_x = identity_block(input_x, 3, [256, 256, 1024])

    # Stage 5
    input_x = convolutional_block(
        input_x,
        middle_window_shape=3,
        filters=[512, 512, 2048],
        s=2,
    )
    input_x = identity_block(input_x, 3, [512, 512, 2048])
    input_x = identity_block(input_x, 3, [512, 512, 2048])

    input_x = tfl.AveragePooling2D(
        pool_size=(2, 2), strides=(2, 2)
    )(input_x)

    input_x = tfl.Flatten()(input_x)
    output_layer = tfl.Dense(
        classes,
        activation="softmax",
        kernel_initializer=glorot_uniform(seed=0)
    )(input_x)

    residual_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    residual_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
        ],
    )
    return residual_model


def run_model(
    num_epochs=100,
    batch_size=32,
    img_width=32,
    img_height=32,
    channels=3,
    save=False,
):
    """
    Service method to run neural net
    :param num_epochs: (int)
    :param batch_size: (int)
    :param img_width: (int)
    :param img_height: (int)
    :param channels: (int)
    :param save: (bool) set True to save compiled tf.model
    :return: (None)
    """

    data_dir = pathlib.Path(
        "{0}/natural_images".format(dogsml.settings.DATA_ROOT)
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode="categorical",
        class_names=dogsml.utils.dataset.NATURAL_IMAGES_CLASS_NAMES,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        label_mode="categorical",
        class_names=dogsml.utils.dataset.NATURAL_IMAGES_CLASS_NAMES,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    print("class_names", class_names)
    classes_count = len(class_names)

    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(
        buffer_size=tf.data.AUTOTUNE
    )

    residual_model = residual_network(
        [img_width, img_height, channels],
        classes_count
    )

    residual_model.fit(
        train_ds,
        epochs=num_epochs,
        validation_data=validation_ds,
    )

    if save:
        residual_model.save("{0}/residual_model_{1}".format(
            dogsml.settings.COMPILED_MODELS_PATH,
            num_epochs)
        )


def run_model_from_h5(
    h5_filename,
    num_epochs=1,
    batch_size=64,
    img_width=64,
    img_height=64,
    channels=3,
    save=False,
):
    """
    Service method to run neural net, read data from hdf5 file
    :param h5_filename: (str)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param img_width: (int)
    :param img_height: (int)
    :param channels: (int)
    :param save: (bool) set True to save compiled tf.model
    :return: (None)
    """

    dataset = h5py.File(h5_filename, "r")
    x_data = np.array(dataset["x_data"])
    y_data = np.array(dataset["y_data"])

    examples = x_data.shape[0]
    classes_count = y_data.shape[1]

    shuffler = np.random.permutation(examples)
    x_data = x_data[shuffler]
    y_data = y_data[shuffler]

    test_part = examples // 10

    x_test = x_data[:test_part]
    y_test = y_data[:test_part]

    x_train = x_data[test_part:]
    y_train = y_data[test_part:]

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    ).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)
    ).batch(batch_size)

    residual_model = residual_network(
        [img_width, img_height, channels],
        classes_count
    )

    residual_model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=test_dataset,
    )

    if save:
        residual_model.save("{0}/residual_h5_{1}".format(
            dogsml.settings.COMPILED_MODELS_PATH,
            num_epochs)
        )


if __name__ == '__main__':
    run_model_from_h5(
        os.path.join(dogsml.settings.DATA_ROOT, "natural_images.hdf5"),
        num_epochs=2,
        save=True,
        batch_size=64,
    )
