import random
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

import pickle as pk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import networks

# Some code taken from https://machinelearningmastery.com/building-a-regression-model-in-pytorch/

device = torch.device("cpu")


def plot_2D_results_datasizes(train, test, data_sizes, title):
    sizes = [i for i in range(1, len(data_sizes) + 1)]
    epochs = [i for i in range(1, len(train[0]) + 1)]

    v_min = 0
    v_max = 10

    E, N = np.meshgrid(epochs, sizes)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
    fig.tight_layout(pad=5)

    contf1_ = ax[0].contourf(E, N, train, levels=400, cmap=plt.cm.CMRmap)
    ax[0].set_title("Training Loss")
    ax[0].set_yticks(sizes, data_sizes)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Data set size")

    contf2_ = ax[1].contourf(E, N, test, levels=400, cmap=plt.cm.CMRmap)
    ax[1].set_title("Test Loss")
    ax[1].set_yticks(sizes, data_sizes)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Data set size")

    fig.colorbar(contf1_, ax=ax[0])
    fig.colorbar(contf2_, ax=ax[1])

    fig.suptitle("Train and test errors for increasing data set sizes.\n" + title)
    plt.subplots_adjust(top=0.80)
    plt.show()


def get_weights(model):
    layer_weights = {}

    for name, layer in model.named_modules():
        if type(layer) is nn.Linear and name.startswith("lin"):
            layer_weights[name] = []
            for w in layer.weight.tolist():
                for k in w:
                    layer_weights[name].append(k)

    return layer_weights


def init_normal(module):
    if type(module) is nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)


def add_noise(arr, delta):
    v = (np.var(arr) / np.abs(np.max(arr) - np.min(arr))) * delta
    noise = np.random.normal(0, v, arr.shape)
    arr += noise
    return arr


def get_data(data_set, data_set_size, delta_noise):
    data = pd.read_csv(data_set).sample(frac=1)
    x = np.array(data.drop("val", axis=1))
    y = np.array(data["val"])

    x = x[0:data_set_size]
    y = y[0:data_set_size]

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True)

    train_y = add_noise(train_y, delta_noise)
    test_y = add_noise(test_y, delta_noise)

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).reshape(-1, 1).to(device)

    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y, dtype=torch.float32).reshape(-1, 1).to(device)

    return train_x, test_x, train_y, test_y


def get_data_emc(data_set, data_set_size, delta_noise, data_start):
    data = pd.read_csv(data_set)
    x = np.array(data.drop("val", axis=1))
    y = np.array(data["val"])

    x = x[data_start: data_start + data_set_size]
    y = y[data_start: data_start + data_set_size]

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True)

    train_y = add_noise(train_y, delta_noise)
    test_y = add_noise(test_y, delta_noise)

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    train_x = torch.tensor(train_x, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).reshape(-1, 1).to(device)

    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y, dtype=torch.float32).reshape(-1, 1).to(device)

    return train_x, test_x, train_y, test_y


def train_one_emc(model, data_set, data_set_size, delta_noise,
                  minibatch_size, learning_rate, betas_adam,
                  n_epochs, data_start):
    train_x, test_x, train_y, test_y = get_data_emc(data_set, data_set_size, delta_noise, data_start)

    model.apply(init_normal)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=0, lr=learning_rate, betas=betas_adam)

    history = {
        "val_loss": [],
        "train_loss": [],
        "title": ""
    }

    batch_start = torch.arange(0, len(train_x), minibatch_size)
    batches = []
    for start in batch_start:
        x_batch = train_x[start:start + minibatch_size].to(device)
        y_batch = train_y[start:start + minibatch_size].to(device)
        batches.append((x_batch, y_batch))

    with tqdm.tqdm(range(1, n_epochs + 1), ncols=120, colour="green") as bar:
        bar.set_description(f"Training model {type(model).__name__} on size {data_set_size}")

        for epoch in bar:
            model.train()

            for (batch_x, batch_y) in batches:
                pred_y = model.forward(batch_x)
                loss = loss_fn(pred_y, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            history["train_loss"].append(loss.item())

            # Validation
            model.eval()
            y_pred = model(test_x)
            val_loss = loss_fn(y_pred, test_y)
            history["val_loss"].append(val_loss.item())

            bar.set_postfix({"Train": loss.item(), "Test": val_loss.item()})

    return history


def train_one(model, data_set, data_set_size, delta_noise,
              minibatch_size, learning_rate, betas_adam,
              n_epochs, plot_history, architecture):
    train_x, test_x, train_y, test_y = get_data(data_set, data_set_size, delta_noise)
    model.apply(init_normal)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=0, lr=learning_rate, betas=betas_adam)

    history = {
        "val_loss": [],
        "train_loss": [],
        "first_moment": [],
        "second_moment": [],
        "weights_per_layer_and_epoch": [],
        "epoch_saved": [],
        "diff_func_per_epoch": [],
        "x1": train_x[:, 0],
        "x2": train_x[:, 1],
        "title": ""
    }

    weight_cut = int(n_epochs / 9)

    batch_start = torch.arange(0, len(train_x), minibatch_size)
    batches = []
    for start in batch_start:
        x_batch = train_x[start:start + minibatch_size].to(device)
        y_batch = train_y[start:start + minibatch_size].to(device)
        batches.append((x_batch, y_batch))

    # plt.ion()
    # graph = plt.plot([], [])[0]
    # plt.xscale("log")
    # plt.grid()

    with tqdm.tqdm(range(1, n_epochs + 1), ncols=100, colour="green") as bar:
        bar.set_description(f"Training model {type(model).__name__}")

        for epoch in bar:
            model.train()

            for (batch_x, batch_y) in batches:
                pred_y = model.forward(batch_x)
                loss = loss_fn(pred_y, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            first_mom = np.average(
                [float(torch.norm(val["exp_avg"])) for val in optimizer.state_dict()["state"].values()]
            )
            second_mom = np.average(
                [float(torch.norm(val["exp_avg_sq"])) for val in optimizer.state_dict()["state"].values()]
            )

            history["train_loss"].append(loss.item())
            history["first_moment"].append(first_mom)
            history["second_moment"].append(second_mom)

            # Validation
            model.eval()
            y_pred = model(test_x)
            val_loss = loss_fn(y_pred, test_y)
            history["val_loss"].append(val_loss.item())

            if epoch % weight_cut == 0:
                history["weights_per_layer_and_epoch"].append(get_weights(model))
                history["epoch_saved"].append(epoch)

                train_y_pred = model(train_x)
                diff = train_y - train_y_pred
                diff = [i[0] for i in diff.tolist()]

                history["diff_func_per_epoch"].append(diff)

            # if epoch > 1:
            #     graph.remove()
            #     graph = plt.plot(range(1, epoch + 1), history["train_loss"], color="g")[0]
            #     graph = plt.plot(range(1, epoch + 1), history["val_loss"], color="r")[0]
            #     plt.xlim(1, epoch)
            #     plt.pause(0.0000001)

            bar.set_postfix({"Train": loss.item(), "Test": val_loss.item()})

    if plot_history:
        history["title"] = (f"Network training stats. Architecture = {type(model).__name__} with {architecture}, "
                            f"epochs = {n_epochs}, batch size = {minibatch_size}, \n delta_noise = {delta_noise}, "
                            f"data set size = {data_set_size / 1000}k, learning rate = {learning_rate}, "
                            f"betas = {betas_adam}")

        network_to_file(history)

    return history


def test_emc():
    n_epochs = 4000
    minibatch_size = 10
    delta_noise = 1
    data_set = "data/emc_data/data5var_k=2.806_50k"
    learning_rate = 5e-3
    betas_adam = (0.9, 0.999)
    architecture = [8, 12, 7]
    results_train = []
    results_test = []
    data_sizes = range(50, 501, 50)

    for size in data_sizes:
        one_train_avg = []
        one_test_avg = []
        for step in range(0, 10):
            model = networks.NetSmall(architecture).to(device)

            data_start = size * step

            history = train_one_emc(model, data_set, size, delta_noise,
                                    minibatch_size, learning_rate, betas_adam,
                                    n_epochs, data_start)

            one_train_avg.append(np.mean(history["train_loss"]))
            one_test_avg.append(np.mean(history["val_loss"]))

        results_train.append(np.mean(one_train_avg))
        results_test.append(np.mean(one_test_avg))

    title = (f"Network parameters: epochs = {n_epochs}, \n Architecture = {type(model).__name__} with {architecture}"
             f", batch size = {minibatch_size}, delta_noise = {delta_noise}, \n"
             f"learning rate = {learning_rate}, betas = {betas_adam}.")

    network_to_file({
        "train_results": results_train,
        "test_results": results_test,
        "data_sizes": data_sizes
    })

    plt.figure(figsize=(10, 7))
    plt.plot(data_sizes, results_train, label="Train")
    plt.plot(data_sizes, results_test, label="Test")
    plt.xlabel("Data set size.")
    plt.ylabel("Expected average error.")
    plt.legend()
    plt.title(title, pad=10)
    plt.show()


def modelwise_dd(data_set, data_set_size, delta_noise, minibatch_size,
                 learning_rate, betas_adam, n_epochs):

    results_train = []
    results_test = []

    models = [[5, 5, 5], [6, 8, 7], [8, 12, 7], [10, 14, 8], [12, 16, 9], [13, 20, 11],
              [14, 27, 12], [17, 32, 14], [21, 38, 17], [25, 44, 19], [30, 50, 20],
              [37, 62, 23], [50, 75, 35], [80, 130, 60]]

    for architecture in models:
        print("Training with architecture: ", architecture)
        a, b = [], []
        for step in range(0, 10):
            model = networks.NetSmall(architecture).to(device)

            data_start = data_set_size * step
            print(f"Training on data from {data_start} to {data_start + data_set_size}")
            history = train_one_emc(model, data_set, data_set_size, delta_noise,
                                    minibatch_size, learning_rate, betas_adam,
                                    n_epochs, data_start)

            a.append(history["train_loss"][-1])
            b.append(history["val_loss"][-1])

        results_train.append(np.mean(a))
        results_test.append(np.mean(b))

    title = (f"Network parameters: Architecture = {type(model).__name__},  epochs = {n_epochs}, "
             f"batch size = {minibatch_size}, \n delta_noise = {delta_noise}, data set size = {data_set_size / 1000}k, "
             f"learning rate = {learning_rate}, betas = {betas_adam}.")

    network_to_file({
        "train_results": results_train,
        "test_results": results_test,
        "models": models
    })

    x = range(1, len(models) + 1)
    plt.figure(figsize=(8, 8))
    plt.plot(x, results_test, label="Test loss")
    plt.plot(x, results_train, label="Train loss")
    plt.xlabel("Model")
    plt.ylabel("Loss")
    plt.title(title, pad=20)
    models = [str(m) for m in models]
    plt.xticks(x, models, rotation= -50)
    plt.grid()
    plt.legend()
    plt.show()


def network_to_file(history):
    name = datetime.now().strftime("%d%m%Y_%H%M%S")

    filename = f"data/produced_files/from_{name}.txt"
    with open(filename, "wb") as file:
        pk.dump(history, file)
        print(f"Wrote history to: " + filename + ".")


def main():
    # torch.manual_seed(0)
    # np.random.seed(0)

    n_epochs = 8000
    minibatch_size = 10

    # Data set settings
    delta_noise = 2
    data_set = "data/data_set/data5var_k=2.806_50k"
    data_set_size = 300

    # Adam parameters
    learning_rate = 5e-4
    betas_adam = (0.9, 0.999)

    architecture = [43, 90, 30]
    # architecture = [9, 12, 16, 13, 11, 9, 7]
    # architecture = [7, 11, 15, 21, 17, 15, 12, 10, 8, 7, 6]

    model = networks.NetSmall(architecture).to(device)
    # model = networks.NetMedium(architecture).to(device)
    # model = networks.NetLarge(architecture).to(device)

    # test_data(model, data_set_size, delta_noise, device,
    #          minibatch_size, learning_rate, betas_adam, n_epochs)

    # modelwise_dd(data_set, data_set_size, delta_noise,
    #             minibatch_size, learning_rate, betas_adam, n_epochs)

    his = train_one(model, data_set, data_set_size, delta_noise,
                    minibatch_size, learning_rate, betas_adam,
                    n_epochs, True, architecture)

    # test_emc()


if __name__ == '__main__':
    main()
