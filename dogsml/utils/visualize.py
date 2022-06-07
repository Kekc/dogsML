import matplotlib.pyplot as plt
import numpy as np


def visualize_history(history, num_epochs):
    """
    Visualize training process of the neural net
    :param history: (tf.History) Result of tf.Model.fit()
    :param num_epochs: (int)
    :return: (None)
    """
    fig, ax = plt.subplots()
    ax.plot(history.history["loss"], label="Train Loss")
    ax.plot(history.history["val_loss"], label="Test Loss")
    ax.legend(loc="upper right")
    plt.ylabel("Loss value")
    plt.xlabel("Epochs")
    max_loss = int(
        max(history.history["loss"] + history.history["val_loss"])
    ) + 1
    plt.axis([0, num_epochs - 1, 0, max_loss])
    plt.show()

    train_accuracy = np.array(history.history["accuracy"]) * 100
    test_accuracy = np.array(history.history["val_accuracy"]) * 100
    fig, ax = plt.subplots()
    ax.plot(
        train_accuracy,
        label="Train Accuracy %",
    )
    ax.plot(
        test_accuracy,
        label="Test Accuracy %",
    )
    ax.legend(loc="lower right")
    plt.ylabel("Accuracy value %")
    plt.xlabel("Epochs")
    plt.axis([0, num_epochs - 1, 0, 100])
    plt.show()
