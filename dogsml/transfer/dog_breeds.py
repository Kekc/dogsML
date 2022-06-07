import os

import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam

import dogsml.settings
import dogsml.utils


def run_model(
    data_dir,
    img_height=160,
    img_width=160,
    batch_size=32,
    validation_split=0.2,
    seed=123,
    num_epochs=1,
    save=False,
    visualize_flag=False,
):
    """
        Service method to run neural net, read data from hdf5 file
        :param data_dir: (str) Path to directory with images
            example structure:
                data_dir
                 - class_A
                 - class_B
                 ...
        :param img_height: (int)
        :param img_width: (int)
        :param batch_size: (int)
        :param validation_split: (float) how much data to set apart for
            validation interval [0; 1]
        :param seed: (int) seed value for correct validation splitting
        :param num_epochs: (int)
        :param save: (bool) set True to save compiled tf.model
        :param visualize_flag: (bool) set True to display plots
        :return: (None)
        """

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    input_shape = (img_height, img_width, 3)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    data_augmentation = tf.keras.Sequential([
        tfl.RandomFlip("horizontal"),
        tfl.RandomRotation(0.2),
    ])

    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)

    x_data = data_augmentation(inputs)
    x_data = tf.keras.applications.mobilenet_v2.preprocess_input(x_data)

    x_data = base_model(x_data, training=False)
    x_data = tfl.GlobalAveragePooling2D()(x_data)
    x_data = tfl.Dropout(0.2)(x_data)
    outputs = tfl.Dense(
        len(class_names),
        activation="softmax",
        kernel_initializer=glorot_uniform(seed=0)
    )(x_data)

    transfer_model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 3 * 10**-3
    transfer_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_top = transfer_model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=num_epochs,
    )

    base_model = transfer_model.layers[4]
    base_model.trainable = True
    fine_tune_at = 135
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    transfer_model.compile(
        optimizer=Adam(learning_rate=10**-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_base = transfer_model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=num_epochs * 2,
        initial_epoch=history_top.epoch[-1]
    )
    if visualize_flag:
        dogsml.utils.visualize.visualize_history(history_top, num_epochs * 2)
        dogsml.utils.visualize.visualize_history(history_base, num_epochs * 2)

    if save:
        transfer_model.save("{0}/dog_breeds_{1}.h5".format(
            dogsml.settings.COMPILED_MODELS_PATH,
            num_epochs,
        ), save_format="h5")


if __name__ == '__main__':
    data_dir = os.path.join(
        dogsml.settings.DATA_ROOT,
        "dog_breeds/images/Images",
    )
    run_model(
        data_dir,
        img_height=224,
        img_width=224,
        batch_size=32,
        num_epochs=5,
        validation_split=0.2,
        save=True,
    )
