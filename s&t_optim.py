import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from termcolor_dg import colored

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import networks_width

# "cuda" for gpu
device = torch.device("cpu")


class FakeData(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


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
    data = pd.read_csv(data_set)
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

    return train_x, test_x, train_y, test_y


def plot_network_history(history, title, epoch_saved, weights_per_layer_epoch):
    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(4, 3, figsize=(24, 20))
    fig.set_layout_engine()
    fig.suptitle(title)
    fig.tight_layout(pad=3.5)

    # Plot history
    axs[0][0].plot(history["val_loss"], label="val_loss")
    axs[0][0].plot(history["train_loss"], label="train_loss")
    axs[0][0].set_ylabel("MSE")
    axs[0][0].set_xlabel("Epoch")
    axs[0][0].legend()

    axs[0][1].plot(history["first_moment"], color="red")
    axs[0][1].set_title("First moment")
    axs[0][1].set_ylabel("Norm")
    axs[0][1].set_xlabel("Epoch")

    axs[0][2].plot(history["second_moment"], color="magenta")
    axs[0][2].set_title("Second moment")
    axs[0][2].set_ylabel("Norm")
    axs[0][2].set_xlabel("Epoch")

    # Plot weight distribution per layer
    flattened_axs = [item for row in axs[1:] for item in row]

    for (subplt, dct, ep) in zip(flattened_axs, weights_per_layer_epoch, epoch_saved):

        for name in dct.keys():
            subplt.hist(dct[name], histtype="bar", bins=25, label=name, alpha=0.5)
        subplt.set_title(f"Epoch {ep}")
        subplt.legend()

    fig.text(0.5, 0.02, "Weight value", ha="center", va="center")
    fig.text(0.015, 0.35, "Number of occurrences", ha="center", va="center", rotation="vertical")

    plt.show()


def set_and_train():

    # Model
    n_epochs = 180
    minibatch_size = 10
    model = networks_width.NetW16().to(device)

    # Data set settings
    delta_noise = 1
    data_file = "data/data_v2/data5var_k=3.204_20k"
    data_set_size = 5000

    # Adam parameters
    learning_rate = 1e-3
    betas_adam = (0.9, 0.999)

    df = pd.read_csv(data_file)
    train_x, test_x, train_y, test_y = get_data(data_file, data_set_size, delta_noise)

    test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y, dtype=torch.float32).reshape(-1, 1).to(device)

    train_data = FakeData(train_x, train_y)
    loader = DataLoader(train_data, shuffle=True, batch_size=minibatch_size)


    model.apply(init_normal)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=0, lr=learning_rate, betas=betas_adam)

    epoch_saved = []
    weights_per_layer_epoch = []
    history = {
        "val_loss": [],
        "train_loss": [],
        "first_moment": [],
        "second_moment": []
    }

    weight_cut = int(n_epochs / 9)

    print(colored(f"Training model {type(model).__name__}.", (255, 20, 20)))

    for epoch in range(1, n_epochs + 1):
        model.train()
        t_start = time.time()

        for batch_x, batch_y in loader:
            pred_y = model.forward(batch_x)
            loss = loss_fn(pred_y, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t_end = time.time()
        batch_per_sec = (data_set_size / minibatch_size) / (t_end - t_start)

        text = f"[ Epoch {epoch} | {batch_per_sec} batch/s | MSE = {loss.item()} ]"
        print(colored(text, (0, 255, 0)))

        first_mom = np.average(
            [float(torch.norm(val["exp_avg"])) for val in optimizer.state_dict()["state"].values()]
        )
        second_mom = np.average(
            [float(torch.norm(val["exp_avg_sq"])) for val in optimizer.state_dict()["state"].values()]
        )

        if (epoch) % weight_cut == 0:
            weights_per_layer_epoch.append(get_weights(model))
            epoch_saved.append(epoch)

        history["train_loss"].append(loss.item())
        history["first_moment"].append(first_mom)
        history["second_moment"].append(second_mom)

        # Validation
        model.eval()
        y_pred = model(test_x)
        val_loss = loss_fn(y_pred, test_y)
        history["val_loss"].append(val_loss.item())

    title = (f"Network training stats. Architecture = {type(model).__name__},  epochs = {n_epochs}, "
                 f"batch size = {minibatch_size}, \n delta_noise = {delta_noise}, data set size = {data_set_size / 1000}k,"
                 f"learning rate = {learning_rate}, betas = {betas_adam}")

    plot_network_history(history, title, epoch_saved, weights_per_layer_epoch)


if __name__ == '__main__':
    set_and_train()
