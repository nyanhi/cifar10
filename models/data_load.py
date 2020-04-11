# coding=utf-8
import numpy as np
from tensorflow import keras


def cifar10_dataload():
    """

    :return:
    """

    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train, y_train = X_train_full[5000:], y_train_full[5000:]
    X_valid, y_valid = X_train_full[:5000], y_train_full[:5000]

    return X_train, y_train, X_valid, y_valid, X_test, y_test
