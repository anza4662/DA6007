import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Normalization, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
import tensorflow.keras.metrics as metrics


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 600])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Critical Temperature]')
    plt.legend()
    plt.grid(True)


def main():

    predictor = "critical_temp"
    data_df = pd.read_csv("train.csv")

    print(data_df.describe().transpose())

    train, test = train_test_split(data_df, test_size=0.50)

    train_X = train.drop(predictor, axis=1)
    train_y = np.array(train[predictor])

    test_X = test.drop(predictor, axis=1)
    test_y = np.array(test[predictor])

    normalizer = Normalization(axis=-1)
    normalizer.adapt(np.array(train_X))

    model = Sequential([
        normalizer,
        Dense(160, activation="relu"),
        Dense(320, activation="relu"),
        Dense(160, activation="relu"),
        Dense(80, activation="relu"),
        Dense(40, activation="relu"),
        Dense(20, activation="relu"),
        Dense(10, activation="relu"),
        Dense(1)
    ])

    model.compile(loss="mean_squared_error", optimizer="adam")
    history = model.fit(train_X, train_y, epochs=300, validation_split=0.2)

    plot_loss(history)
    plt.show()


if __name__ == '__main__':
    main()
