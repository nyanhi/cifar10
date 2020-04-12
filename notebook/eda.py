import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

n_rows = 4
n_cols = 10

label_names = np.array([
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
])

plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train_full[index])
        plt.axis('off')
        plt.title(label_names[y_train_full[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.savefig('cifar10-sample.png')
plt.show()
