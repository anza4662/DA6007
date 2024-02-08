import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Normalization, Dense, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback


class WeightCapture(Callback):
    "Capture the weights of each layer of the model"

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.weights = []
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        self.epochs.append(epoch)  # remember the epoch axis
        weight = {}
        for layer in self.model.layers[1:]:
            if not layer.weights:
                continue
            name = layer.weights[0].name.split("/")[0]
            weight[name] = layer.weights[0].numpy()
        self.weights.append(weight)


def weight_histogram(model):
    histo = dict()

    for w in model.get_weights():
        print(type(w))
        for w2 in w:
            temp = int(int(w2 * 100) / 100)
            if temp not in histo:
                histo[temp] = 0

            histo[temp] += 1


def plot_weights_over_epochs(capture_cb):
    "Plot the weights' mean and s.d. across epochs"
    fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True, figsize=(8, 10))
    ax[0].set_title("Mean weight")
    for key in capture_cb.weights[0].keys():
        ax[0].plot(capture_cb.epochs, [w[key].mean() for w in capture_cb.weights], label=key)
    ax[0].legend()
    ax[1].set_title("S.D.")
    for key in capture_cb.weights[0].keys():
        ax[1].plot(capture_cb.epochs, [w[key].std() for w in capture_cb.weights], label=key)
    ax[1].legend()
    plt.show()

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

    # print(data_df.describe().transpose())

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
        BatchNormalization(axis=-1),
        Dense(10, activation="relu"),
        Dense(3, activation="relu"),
        Dense(1)
    ])

    model.compile(loss="mean_squared_error", optimizer="adam")
    capture_cb = WeightCapture(model)
    capture_cb.on_epoch_end(-1)
    history = model.fit(train_X, train_y, epochs=5, validation_split=0.2, callbacks=[capture_cb])

    plot_loss(history)
    plot_weights_over_epochs(capture_cb)
    weight_histogram(model)
    plt.show()


if __name__ == '__main__':
    main()
