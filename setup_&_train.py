import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import tqdm

import pickle as pk
import numpy as np
from sklearn.preprocessing import StandardScaler

import networks

# Some code taken from https://machinelearningmastery.com/building-a-regression-model-in-pytorch/

device = torch.device("cuda")

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


def get_data(sample_size, file):
    test_size = int((0.3 * sample_size) / 0.7)
    data = get_data_from_file(file)
    train_x = data["train_x"][:sample_size].to(device)
    train_y = data["train_y"][:sample_size].to(device)
    test_x = data["test_x"][:test_size].to(device)
    test_y = data["test_y"][:test_size].to(device)

    std_noise = data["std_noise"]
    x_range = data["x_range"]
    c = data["c"]
    k = data["k"]

    return train_x, train_y, test_x, test_y, std_noise, k, c, x_range


def train_one(n_epochs, minibatch_size, architecture, model, sample_size, learning_rate, betas, train_x, train_y,
              test_x, test_y, name, std_noise, k, c, x_range):
    model.apply(init_normal)
    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), weight_decay=0, lr=learning_rate, betas=betas)

    history = {
        "model": None,
        "val_loss": [],
        "train_loss": [],
        "first_moment": [],
        "second_moment": [],
        "weights_per_layer_and_epoch": [],
        "model_per_epoch": [],
        "epoch_saved": [],
        "diff_func_per_epoch": [],
        "mean_diff_squared": [],
        "var_diff": [],
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "std_noise": std_noise,
        "architecture": architecture,
        "sample_size": sample_size,
        "minibatch_size": minibatch_size,
        "learning_rate": learning_rate,
        "betas_adam": betas,
        "n_epochs": n_epochs,
        "k": k,
        "c": c,
        "x_range": x_range
    }

    weight_cut = int(n_epochs / 9)
    weight_cuts = [80, 100, 120, 140, 160, 180, 200, 220, 240]

    batch_start = torch.arange(0, len(train_x), minibatch_size)
    batches = []
    for start in batch_start:
        x_batch = train_x[start:start + minibatch_size].to(device)
        y_batch = train_y[start:start + minibatch_size].to(device)
        batches.append((x_batch, y_batch))

    n_batches = len(batches)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    with tqdm.tqdm(range(1, n_epochs + 1), ncols=150, colour="green") as bar:
        bar.set_description(f"Training model {type(model).__name__}")
        for epoch in bar:
            model.train()

            avg_loss_per_epoch = 0
            random.shuffle(batches)

            for (batch_x, batch_y) in batches:
                pred_y = model.forward(batch_x)
                loss = loss_fn(pred_y, batch_y)
                avg_loss_per_epoch += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            first_mom = np.average(
                [float(torch.norm(val["exp_avg"])) for val in optimizer.state_dict()["state"].values()]
            )
            second_mom = np.average(
                [float(torch.norm(val["exp_avg_sq"])) for val in optimizer.state_dict()["state"].values()]
            )

            avg_loss_per_epoch = avg_loss_per_epoch / n_batches

            history["train_loss"].append(avg_loss_per_epoch)
            history["first_moment"].append(first_mom)
            history["second_moment"].append(second_mom)

            # Validation
            model.eval()
            with torch.no_grad():
                y_pred = model(test_x).to(device)
                val_loss = loss_fn(y_pred, test_y)
                history["val_loss"].append(val_loss.item())

                train_y_pred = model(train_x)
                diff = train_y - train_y_pred

                mean_diff_squared = np.mean(diff.tolist()) ** 2
                var_diff = np.var(diff.tolist())

                history["var_diff"].append(var_diff)
                history["mean_diff_squared"].append(mean_diff_squared)

                if epoch in weight_cuts:
                    history["weights_per_layer_and_epoch"].append(get_weights(model))
                    history["epoch_saved"].append(epoch)

                    history["model_per_epoch"].append(copy.deepcopy(model.state_dict()))

                    history["diff_func_per_epoch"].append([i[0] for i in diff.tolist()])

            bar.set_postfix({"Train": avg_loss_per_epoch, "Test": val_loss.item()})

    history["optimizer"] = optimizer.state_dict()
    history["model"] = model.state_dict()

    network_to_file(history, name)


def network_to_file(history, name):
    date_now = datetime.now().strftime("%d%m%Y_%H%M%S")
    filename = f"produced_files/{name}_{date_now}.txt"

    with open(filename, "wb") as file:
        pk.dump(history, file)
        print(f"Wrote history to {filename}.")


def get_data_from_file(filename):
    with open(filename, "rb") as file:
        history = pk.load(file)
        print("Succesfully read data.")

    return history


def continue_training():
    n_epochs = 2000
    path = "/home/anza/kanidiatarbete/dd_tests/50mb/10k_samples FINE/Small/five_var_data_24042024_211127.txt"
    data = get_data_from_file(path)

    model_dict = data["model"]
    model = networks.NetSmall(data["architecture"]).to(device)
    model.load_state_dict(model_dict)

    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(data["optimizer"])
    epoch_start = len(data["val_loss"])

    train_x = data["train_x"]
    train_y = data["train_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]
    minibatch_size = data["minibatch_size"]

    batch_start = torch.arange(0, len(train_x), minibatch_size)
    batches = []
    for start in batch_start:
        x_batch = train_x[start:start + minibatch_size].to(device)
        y_batch = train_y[start:start + minibatch_size].to(device)
        batches.append((x_batch, y_batch))

    n_batches = len(batches)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    with tqdm.tqdm(range(epoch_start + 1, epoch_start + n_epochs + 1), ncols=150, colour="green") as bar:
        bar.set_description(f"Training model {type(model).__name__}")
        for epoch in bar:
            model.train()

            avg_loss_per_epoch = 0
            random.shuffle(batches)

            for (batch_x, batch_y) in batches:
                pred_y = model.forward(batch_x)
                loss = loss_fn(pred_y, batch_y)
                avg_loss_per_epoch += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            first_mom = np.average(
                [float(torch.norm(val["exp_avg"])) for val in optimizer.state_dict()["state"].values()]
            )
            second_mom = np.average(
                [float(torch.norm(val["exp_avg_sq"])) for val in optimizer.state_dict()["state"].values()]
            )

            avg_loss_per_epoch = avg_loss_per_epoch / n_batches

            data["train_loss"].append(avg_loss_per_epoch)
            data["first_moment"].append(first_mom)
            data["second_moment"].append(second_mom)

            # Validation
            model.eval()
            with torch.no_grad():
                y_pred = model(test_x).to(device)
                val_loss = loss_fn(y_pred, test_y)
                data["val_loss"].append(val_loss.item())

                train_y_pred = model(train_x)
                diff = train_y - train_y_pred

                mean_diff_squared = np.mean(diff.tolist()) ** 2
                var_diff = np.var(diff.tolist())

                data["var_diff"].append(var_diff)
                data["mean_diff_squared"].append(mean_diff_squared)

            bar.set_postfix({"Train": avg_loss_per_epoch, "Test": val_loss.item()})

    data["optimizer"] = optimizer.state_dict()
    data["model"] = model.state_dict()

    network_to_file(data, "five_var_data_extended")


def train():
    # Model parameters
    n_epochs = 4000
    minibatch_size = 50
    three_var = False

    architecture = [6, 7, 6]
    # architecture = [8, 12, 16, 13, 11, 9, 7]
    # architecture = [7, 13, 17, 24, 30, 24, 20, 17, 14, 10, 8]
    # architecture = [8, 12, 17, 23, 28, 36, 33, 31, 29, 26, 21, 17, 13, 11, 7]
    # architecture = [8, 12, 16, 24, 29, 34, 39, 43, 47, 53, 60,
    #                67, 75, 80, 74, 65, 57, 51, 45, 40, 35, 27, 22, 18, 14, 11, 7]

    model = networks.NetSmall(architecture).to(device)
    # model = networks.NetMedium(architecture).to(device)
    # model = networks.NetLarge(architecture).to(device)
    # model = networks.NetHuge(architecture).to(device)
    # model = networks.NetExtreme(architecture).to(device)

    # Dataset parameters
    # k = 5.5
    # c = 5
    # x_range = 3
    sample_size = 10000

    # Adam parameters
    learning_rate = 1e-3
    betas = (0.9, 0.99)

    data_set_path = "/home/anza/kanidiatarbete/data/data_five_var_v5.txt"

    if three_var:
        name = "three_var_data"
    else:
        name = "five_var_data"

    train_x, train_y, test_x, test_y, std_noise, k, c, x_range = get_data(sample_size, data_set_path)

    train_one(n_epochs, minibatch_size, architecture, model, sample_size, learning_rate, betas,
              train_x, train_y, test_x, test_y, name, std_noise, k, c, x_range)


if __name__ == '__main__':
    train()
    # continue_training()
