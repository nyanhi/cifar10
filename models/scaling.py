# coding=utf-8


def data_scaling(X_train, X_valid, X_test):
    """
    Normalize data by mean of X_train and stds of X_train
    """
    X_means = X_train.mean(axis=0)
    X_stds = X_train.std(axis=0)

    X_train_scaled = (X_train - X_means) / X_stds
    X_valid_scaled = (X_valid - X_means) / X_stds
    X_test_scaled = (X_test - X_means) / X_stds

    return X_train_scaled, X_valid_scaled, X_test_scaled
