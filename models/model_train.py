import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation

from models.logdir import get_run_logdir
from models.data_load import cifar10_dataload
from models.scaling import data_scaling


keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)


def base_model(n_hidden=20, n_neurons=100, learning_rate=1e-4, input_shape=[32, 32, 3]):
    """
    base model:
    20 Dense layers with 100 neurons - intentionally too many because ----
    Initialized with He normalization
    ELU activation function
    Nadam optimization
    """
    model = keras.models.Sequential()
    model.add(Flatten(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation='elu', kernel_initializer='he_normal'))
    model.add(Dense(10, activation='softmax'))
    optimizer = keras.optimizers.Nadam(learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def bn_model(n_hidden=20, n_neurons=100, learning_rate=1e-4, input_shape=[32, 32, 3]):
    """
    added batch normalization to the base model to compare learning curve.
    """
    model = keras.models.Sequential()
    model.add(Flatten(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation(activation='elu'))
    model.add(Dense(10, activation='softmax'))
    optimizer = keras.optimizers.Nadam(learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def selu_model(n_hidden=20, n_neurons=100, learning_rate=1e-4, input_shape=[32, 32, 3]):
    """
    added batch normalization to the base model to compare learning curve.
    """
    model = keras.models.Sequential()
    model.add(Flatten(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, kernel_initializer='lecun_normal', activation='selu'))
    model.add(Dense(10, activation='softmax'))
    optimizer = keras.optimizers.Nadam(learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == '__main__':

    model_selected = selu_model

    X_train, y_train, X_valid, y_valid, X_test, y_test = cifar10_dataload()

    if model_selected == selu_model:
        X_train, X_valid, X_test = data_scaling(X_train, X_valid, X_test)

    model = model_selected()
    # model.summary()

    # callbacks
    checkpoint_cb = keras.callbacks.ModelCheckpoint('selu_model.h5', save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    run_logdir = get_run_logdir('selu_model')
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    callbacks = [checkpoint_cb, early_stopping_cb, tensorboard_cb]

    # training model

    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=callbacks)

