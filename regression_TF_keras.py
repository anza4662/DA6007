import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Normalization, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Error [Strength]')
    plt.legend()
    plt.grid(True)


def main():
    data_df = pd.read_csv("concrete.csv")
    train, test = train_test_split(data_df, test_size=0.30)

    train_X = train.drop("strength", axis=1)
    train_y = np.array(train["strength"])

    test_X = test.drop("strength", axis=1)
    test_y = np.array(test["strength"])

    # sns.pairplot(train_X[['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer',
    #                      'Coarse Aggregate', 'Fine Aggregate', 'Age (day)']], diag_kind='kde')

    normalizer = Normalization(axis=-1)
    normalizer.adapt(np.array(train_X))

    model = Sequential([
        normalizer,
        Dense(16, activation="relu"),
        Dense(32, activation="relu"),
        Dense(64, activation="relu"),
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(3, activation="relu"),
        Dense(1)
    ])

    model.compile(loss="mean_squared_error", optimizer="adam")
    print(model.summary())
    history = model.fit(train_X, train_y, epochs=5000, verbose=1,
                        validation_split=0.2)

    plot_loss(history)
    plt.show()


if __name__ == '__main__':
    main()
