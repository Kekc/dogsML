import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.initializers import glorot_uniform

import dogsml.settings
import dogsml.utils


def create_transfer_model(img_height, img_width, num_classes):
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    input_shape = (img_height, img_width, 3)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False
    inputs = tf.keras.Input(shape=input_shape)

    x_data = preprocess_input(inputs)

    x_data = base_model(x_data, training=False)
    x_data = tfl.GlobalAveragePooling2D()(x_data)
    x_data = tfl.Dropout(0.2)(x_data)
    outputs = tfl.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer=glorot_uniform(seed=0)
    )(x_data)

    return tf.keras.Model(inputs, outputs)


def run_model(
    data_dir,
    img_height=160,
    img_width=160,
    batch_size=32,
    validation_split=0.2,
    seed=55,
    num_epochs=1,
    save=False,
):

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

    transfer_model = create_transfer_model(
        img_height, img_width, len(class_names)
    )

    base_learning_rate = 0.001
    transfer_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    transfer_model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=num_epochs
    )

    if save:
        transfer_model.save("{0}/transfer_{1}".format(
            dogsml.settings.COMPILED_MODELS_PATH,
            num_epochs,
        ))


if __name__ == '__main__':
    data_dir = "{0}/natural_images".format(dogsml.settings.DATA_ROOT)
    run_model(
        data_dir,
        num_epochs=3,
        save=True,
    )
