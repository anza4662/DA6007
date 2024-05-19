import random

import numpy as np
import pickle as pk

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import torch
from torch import nn
from scipy.interpolate import griddata

import networks


def get_data_from_file(filename):
    with open(filename, "rb") as file:
        history = pk.load(file)
        print("Successfully read data.")
    return history


def plot_training_curve_and_diff(history):
    # fig, axs = plt.subplots(1, 1, figsize=(17, 7))
    # fig.tight_layout(pad=2)
    #
    # axs[0].plot(history["val_loss"], label="Validation loss")
    # axs[0].plot(history["train_loss"], label="Training loss")
    # axs[0].plot([history["std_noise"] for i in range(len(history["train_loss"]))], label="Data noise variance")
    # axs[0].set_ylabel("MSE")
    # axs[0].set_xscale("log")
    # # axs[0].set_ylim([0, 5])
    # axs[0].grid()
    # axs[0].set_xlabel("Epoch")
    # axs[0].legend()

    # axs[1].plot(history["mean_diff_squared"], label="Bias²")
    # axs[1].plot(history["var_diff"], label="Variance")
    # axs[1].plot(history["train_loss"], label="Training loss")
    # axs[1].set_title("Bias variance trade-off.")
    # axs[1].set_ylabel("MSE")
    # axs[1].set_xlabel("Epoch")
    # axs[1].set_xscale("log")
    # # axs[1].set_ylim([0, 5])
    # axs[1].grid()
    # axs[1].legend()

    plt.plot(history["val_loss"], label="Test error")
    plt.plot(history["train_loss"], label="Training error")
    plt.plot([history["std_noise"] for i in range(len(history["train_loss"]))], label="Standard error \n in noise")
    plt.ylabel("Mean Squared Error")
    plt.xscale("log")
    plt.title("Training curve (Extreme network).")
    # plt.ylim([0, 5])
    plt.grid()
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


def plot_moments(history):
    fig, axs = plt.subplots(1, 2, figsize=(17, 7))
    fig.tight_layout(pad=2)

    axs[0].plot(history["first_moment"], color="red")
    axs[0].set_title("First moment")
    axs[0].set_ylabel("Norm")
    axs[0].set_xlabel("Epoch")

    axs[1].plot(history["second_moment"], color="magenta")
    axs[1].set_title("Second moment")
    axs[1].set_ylabel("Norm")
    axs[1].set_xlabel("Epoch")
    plt.show()


def plot_weight_distributions(history):
    fig, axs = plt.subplots(3, 3, figsize=(24, 15))
    fig.tight_layout(pad=3.5)

    flattened_axs = [item for row in axs for item in row]

    for (subplt, dct, ep) in zip(flattened_axs, history["weights_per_layer_and_epoch"], history["epoch_saved"]):

        for name in dct.keys():
            subplt.hist(dct[name], histtype="bar", bins=50, label=name, alpha=0.5)
        subplt.set_title(f"Epoch {ep}")
        subplt.legend()

    fig.text(0.5, 0.02, "Weight value", ha="center", va="center")
    fig.text(0.015, 0.5, "Number of occurrences", ha="center", va="center", rotation="vertical")
    plt.show()

def g(x_n, c, k):
    prod1 = x_n[0] * x_n[2]
    prod2 = x_n[1] * x_n[0] * x_n[3]
    prod3 = x_n[2] * x_n[1] * x_n[4]
    return k * (np.sin((c - k) * prod1) + np.cos((c - k) * prod2) - np.sin((c - k) * prod3))


def plot_curves(history):
    k = history["k"]
    c = history["c"]
    x_range = history["x_range"]
    n = 100

    fig = plt.figure(figsize=(20, 20))
    x = np.linspace(-x_range, x_range, n)

    vals_true = []
    vals_model = []

    architecture = history["architecture"]
    model = networks.NetHuge(architecture).to(torch.device("cuda"))
    model_dict = history["model"]
    model.load_state_dict(model_dict)
    model.eval()

    for i in range(n):
        print(i)
        for j in range(n):
            x_s = [x[i], 1, x[j], 1, 1]

            vals_true.append(g(x_s, c, k))
            x_s = np.array(x_s)

            with torch.no_grad():
                x_s = torch.tensor(np.array([x_s]), dtype=torch.float32).to(torch.device("cuda"))
                vals_model.append(model.forward(x_s).tolist()[0][0])

    vals_true = np.array(vals_true)
    vals_model = np.array(vals_model)
    vals_true = vals_true.reshape(n, n)
    vals_model = vals_model.reshape(n, n)

    xi1, xi2 = np.meshgrid(x, x)

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot_wireframe(xi1, xi2, vals_true, color='green')
    ax1.set_title("True function (x1, x2).")

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot_wireframe(xi1, xi2, vals_model, color='red')
    ax2.set_title("Model prediction (x1, x2).")
    fig.tight_layout(pad=0.05)
    plt.show()


def plot_diff_curve_v2(history):
    fig = plt.figure(figsize=(24, 15))
    x1 = np.array([x[0] for x in history["train_x"]])
    x2 = np.array([x[1] for x in history["train_x"]])

    for (indx, z, ep) in zip(range(1, 10), np.array(history["diff_func_per_epoch"]), history["epoch_saved"]):
        ax = fig.add_subplot(3, 3, indx, projection='3d')
        surf = ax.plot_trisurf(x1, x2, z, cmap=plt.cm.CMRmap)
        ax.set_title(f"Epoch {ep}")
        fig.colorbar(surf, location="left", ax=ax)

    fig.tight_layout(pad=1.2)
    plt.show()


def plot_diff_curve(history):
    fig = plt.figure(figsize=(24, 15))
    x1 = np.array([x[0] for x in history["train_x"]])
    x2 = np.array([x[1] for x in history["train_x"]])

    xi = np.linspace(x1.min(), x1.max(), 500)
    yi = np.linspace(x2.min(), x2.max(), 500)
    xig, yig = np.meshgrid(xi, yi)

    for (indx, z, ep) in zip(range(1, 10), np.array(history["diff_func_per_epoch"]), history["epoch_saved"]):
        subplt = fig.add_subplot(3, 3, indx)
        zi = griddata((x1, x2), z, (xi[None, :], yi[:, None]), method='cubic')
        surf = subplt.contourf(xig, yig, zi, 500, cmap=plt.cm.CMRmap)
        subplt.set_title(f"Epoch {ep}")
        fig.colorbar(surf, location="left", ax=subplt)

    fig.tight_layout(pad=1.5)
    plt.show()


def plot_diff_distribution(history):
    fig = plt.figure(figsize=(24, 17))
    fig.tight_layout(pad=2.5)

    for indx, diff, ep in zip(range(1, 11), history["diff_func_per_epoch"], history["epoch_saved"]):
        plot = plt.subplot(3, 3, indx)
        plot.hist(diff, bins=50)
        plot.set_title(f"Epoch {ep}")
        fig.add_subplot(plot)

    fig.text(0.5, 0.01, "Diff value", ha="center", va="center")
    fig.text(0.005, 0.5, "Number of occurrences", ha="center", va="center", rotation="vertical")

    plt.show()


def plot_dd_curve():
    history1 = get_data_from_file(
        "/home/anza/kanidiatarbete/dd_tests/50mb/5k_samples/small/five_var_data_14052024_122823.txt"
    )

    history2 = get_data_from_file(
        "/home/anza/kanidiatarbete/dd_tests/50mb/5k_samples/medium/five_var_data_14052024_131938.txt"
    )

    history3 = get_data_from_file(
        "/home/anza/kanidiatarbete/dd_tests/50mb/5k_samples/large/five_var_data_14052024_130921.txt"
    )

    history4 = get_data_from_file(
        "/home/anza/kanidiatarbete/dd_tests/50mb/5k_samples/huge/five_var_data_14052024_133813.txt"
    )

    # history5 = get_data_from_file(
    #     "/other_tests/dd3/five_var_data_04052024_013309.txt"
    # )
    plt.plot(history2["val_loss"], label="Medium network")
    plt.plot(history3["val_loss"], label="Large network")
    plt.plot(history4["val_loss"], label="Huge network")
    plt.plot(history1["val_loss"], label="Small network")
    # plt.ylim([0, 6.5])
    plt.ylabel("Test Error")
    plt.xscale("log")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid()
    plt.show()


def print_setup(history):
    setup = (f'SETUP \n'
             f'Architecture {history["architecture"]} \n'
             f'Sample size {history["sample_size"]} \n'
             f'Noise std. {history["std_noise"]} \n'
             f'Minibatch size {history["minibatch_size"]} \n'
             f'Learning rate {history["learning_rate"]} \n'
             f'Betas Adam {history["betas_adam"]} \n'
             f'k, c, x_range {history["k"]}, {history["c"]}, {history["x_range"]}')

    print(setup)


def getdatainfo():
    data = get_data_from_file(
        "/home/anza/kanidiatarbete/data/data_five_var_v4.txt")
    print(data)


def get_weights(model):
    layer_weights = {}

    for name, layer in model.named_modules():
        if type(layer) is nn.Linear and name.startswith("lin"):
            layer_weights[name] = []
            for w in layer.weight.tolist():
                for k in w:
                    layer_weights[name].append(k)

    return layer_weights


def plot_extra_weights(history):
    model_dict = history["model"]
    model = networks.NetSmall(history["architecture"])
    model.load_state_dict(model_dict)

    weights_extra = get_weights(model)

    for weight in weights_extra.keys():
        plt.hist(weights_extra[weight], histtype="bar", bins=50, label=weight, alpha=0.5)

    plt.legend()
    plt.show()


def main():
    plt.rcParams.update({'font.size': 15})

    file = "/home/anza/kanidiatarbete/produced_files/five_var_data_17052024_155319.txt"
    history = get_data_from_file(file)

    print("Plotting...")

    # getdatainfo()
    # plot_dd_curve()

    # print_setup(history)
    # plot_curves(history)

    plot_training_curve_and_diff(history)
    # plot_diff_curve(history)
    # plot_moments(history)
    # plot_weight_distributions(history)
    # plot_extra_weights(history)
    # plot_diff_distribution(history)

    print("DONE.")


if __name__ == '__main__':
    main()
