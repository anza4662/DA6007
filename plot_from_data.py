import numpy as np
import pickle as pk

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.interpolate import griddata


def get_data_from_file(filename):
    with open(filename, "rb") as file:
        history = pk.load(file)
        print("Succesfully read data.")
    return history


def plot_training_curve_and_moments(history):
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(history["title"])
    fig.tight_layout(pad=3.5)

    axs[0].plot(history["val_loss"], label="val_loss")
    axs[0].plot(history["train_loss"], label="train_loss")
    axs[0].set_xscale("log")
    axs[0].set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1k"])
    axs[0].set_ylabel("MSE")
    # axs[0].set_ylim(0, 8)
    axs[0].set_title("Training and validation curve.")
    axs[0].grid()
    axs[0].set_xlabel("Epoch")
    axs[0].legend()

    axs[1].plot(history["first_moment"], color="red")
    axs[1].set_title("First moment")
    axs[1].set_ylabel("Norm")
    axs[1].set_xlabel("Epoch")

    axs[2].plot(history["second_moment"], color="magenta")
    axs[2].set_title("Second moment")
    axs[2].set_ylabel("Norm")
    axs[2].set_xlabel("Epoch")
    plt.show()


def plot_emc(history):
    plt.figure(figsize=(10, 7))
    plt.plot(history["data_sizes"], history["train_results"], label="Train")
    plt.plot(history["data_sizes"], history["test_results"], label="Test")
    plt.xlabel("Data set size.")
    plt.ylabel("Expected average error.")
    plt.legend()
    plt.title("Varying datasizes for a small net.", pad=10)
    plt.show()


def plot_weight_distributions(history):
    fig, axs = plt.subplots(3, 3, figsize=(24, 15))
    fig.suptitle(history["title"])
    fig.tight_layout(pad=3.5)

    flattened_axs = [item for row in axs for item in row]

    for (subplt, dct, ep) in zip(flattened_axs, history["weights_per_layer_and_epoch"], history["epoch_saved"]):

        for name in dct.keys():
            subplt.hist(dct[name], histtype="bar", bins=25, label=name, alpha=0.5)
        subplt.set_title(f"Epoch {ep}")
        subplt.legend()

    fig.text(0.5, 0.02, "Weight value", ha="center", va="center")
    fig.text(0.015, 0.5, "Number of occurrences", ha="center", va="center", rotation="vertical")

    plt.show()


def plot_diff_curve(history):
    fig = plt.figure(figsize=(24, 15))
    fig.suptitle(history["title"])
    x1 = np.array(history["x1"])
    x2 = np.array(history["x2"])

    xi = np.linspace(x1.min(), x1.max(), 100)
    yi = np.linspace(x2.min(), x2.max(), 100)
    xig, yig = np.meshgrid(xi, yi)

    for (indx, z, ep) in zip(range(1, 10), np.array(history["diff_func_per_epoch"]), history["epoch_saved"]):
        subplt = fig.add_subplot(3, 3, indx)
        zi = griddata((x1, x2), z, (xi[None, :], yi[:, None]), method='cubic')
        surf = subplt.contourf(xig, yig, zi, 100, cmap=plt.cm.CMRmap)
        subplt.set_title(f"Epoch {ep}")
        fig.colorbar(surf, location="left", ax=subplt)

    fig.tight_layout(pad=1.5)
    plt.show()


def plot_diff_distribution(history):
    fig = plt.figure(figsize=(24, 18))
    fig.suptitle(history["title"])

    for (indx, z, ep) in zip(range(1, 10), np.array(history["diff_func_per_epoch"]), history["epoch_saved"]):
        subplt = fig.add_subplot(3, 3, indx)
        hist = subplt.hist(z, bins=100)
        subplt.set_title(f"Epoch {ep}")

    fig.tight_layout(pad=2.5)
    fig.text(0.5, 0.015, "y_pred - y_true value", ha="center", va="center")
    fig.text(0.005, 0.5, "Number of occurrences", ha="center", va="center", rotation="vertical")
    plt.show()


def main():
    plt.rcParams.update({'font.size': 15})
    history = get_data_from_file("figures/week11/small_net_tests/[6,8,7]/from_13032024_191916.txt")

    # plot_emc(history)

    print("Plotting...")

    # plot_diff_distribution(history)
    plot_diff_curve(history)
    plot_training_curve_and_moments(history)
    plot_weight_distributions(history)

    print("DONE.")


if __name__ == '__main__':
    main()
