import numpy as np
import pickle as pk

import matplotlib.pyplot as plt


def get_data_from_file(filename):
    with open(filename) as file:
        history = pk.load(file)
        print("Succesfully read data.")
    return history


def plot_training_curve_and_moments(history):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.set_suptitle(history["title"])
    fig.tight_layout(pad=3.5)

    axs[0].plot(history["val_loss"], label="val_loss")
    axs[0].plot(history["train_loss"], label="train_loss")
    axs[0].set_ylabel("MSE")
    axs[0].grid()
    axs[0].set_ylim(0, 10)
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


def plot_weight_distributions(history):
    fig, axs = plt.subplots(3, 3, figsize=(24, 15))
    fig.set_suptitle(history["title"])
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
    fig, axs = plt.subplots(3, 3, figsize=(24, 15))
    fig.set_suptitle(history["title"])
    fig.tight_layout(pad=3.5)

    v_min = 0
    v_max = 10

    x1, x2 = np.meshgrid(history["x1"], history["x2"])

    flattened_axs = [item for row in axs for item in row]

    for (subplt, y, ep) in zip(flattened_axs, history["diff_func_per_epoch"], history["epoch_saved"]):

        contf_ = subplt.contourf(x1, x2, y, levels=400, vmin=v_min, vmax=v_max)
        subplt.set_xlabel("x1")
        subplt.set_ylabel("x2")
        subplt.set_title(f"Epoch {ep}")
        fig.colorbar(contf_, ax=subplt)

    plt.show()


def main():
    print("Plotting...")
    plt.rcParams.update({'font.size': 15})

    print("DONE.")


if __name__ == '__main__':
    main()
