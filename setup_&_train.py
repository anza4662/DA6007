import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy
import networks

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.layout_engine as le


# Some code taken from https://machinelearningmastery.com/building-a-regression-model-in-pytorch/


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


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # dev = "cuda" or dev = "cpu"
    dev = "cpu"
    device = torch.device(dev)

    n_epochs = 900
    minibatch_size = 50

    # Data set settings
    delta_noise = 1
    data_set = "data/data5features_0to20_50k"
    data_set_size = 10000

    # Adam parameters
    learning_rate = 1e-1
    betas_adam = (0.9, 0.999)

    # model
    model = networks.Net6().to(device)

    ylim = False
    ylim_train_curve = 0.1

    data = pd.read_csv(data_set)
    X = np.array(data.drop("val", axis=1))
    y = np.array(data["val"])

    if data_set_size != 500000:
        X = X[0:data_set_size]
        y = y[0:data_set_size]

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, shuffle=True)

    train_y = add_noise(train_y, delta_noise)

    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).reshape(-1, 1).to(device)

    test_X = torch.tensor(test_X, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y, dtype=torch.float32).reshape(-1, 1).to(device)

    model.apply(init_normal)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=0, lr=learning_rate, betas=betas_adam)
    batch_start = torch.arange(0, len(train_X), minibatch_size)

    epoch_saved = []
    weights_per_layer_epoch = []
    best_mse = np.inf
    best_weights = None
    history = {
        "val_loss": [],
        "train_loss": [],
        "grad_norm": [],
        "first_moment": [],
        "second_moment": []
    }

    weight_cut = int(n_epochs / 9)

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit=" batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}")

            for start in bar:
                X_batch = train_X[start:start + minibatch_size].to(device)
                y_batch = train_y[start:start + minibatch_size].to(device)

                y_pred = model.forward(X_batch)

                loss = loss_fn(y_pred, y_batch)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                bar.set_postfix(mse=float(loss))

        grad_norm = np.sqrt(sum([(torch.norm(p.grad) ** 2).tolist() for p in model.parameters()]))

        first_mom = np.average(
            [float(torch.norm(val["exp_avg"])) for val in optimizer.state_dict()["state"].values()]
        )
        second_mom = np.average(
            [float(torch.norm(val["exp_avg_sq"])) for val in optimizer.state_dict()["state"].values()]
        )

        if (epoch + 1) % weight_cut == 0:
            weights_per_layer_epoch.append(get_weights(model))
            epoch_saved.append(epoch + 1)

        history["train_loss"].append(loss.item())
        history["grad_norm"].append(grad_norm)
        history["first_moment"].append(first_mom)
        history["second_moment"].append(second_mom)

        # Validation
        model.eval()
        y_pred = model(test_X)
        val_mse = loss_fn(y_pred, test_y)
        val_mse = float(val_mse)
        history["val_loss"].append(val_mse)
        if val_mse < best_mse:
            best_mse = val_mse
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)

    print("Best test error: %.2f" % best_mse)

    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(4, 3, figsize=(24, 20))
    fig.set_layout_engine()
    fig.suptitle(f"Network training stats. Architecture = {type(model).__name__},  epochs = {n_epochs}, "
                 f"batch size = {minibatch_size}, \n delta_noise = {delta_noise}, data set size = {data_set_size / 1000}k, "
                 f"learning rate = {learning_rate}, betas = {betas_adam}")
    fig.tight_layout(pad=3.5)

    # Plot history
    axs[0][0].plot(history["val_loss"], label="val_loss")
    axs[0][0].plot(history["train_loss"], label="train_loss")
    axs[0][0].set_ylabel("MSE")
    axs[0][0].set_xlabel("Epoch")

    if ylim:
        axs[0][0].set_ylim([0, ylim_train_curve])
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
    fig.text(0.02, 0.34, "Number of occurrences", ha="center", va="center", rotation="vertical")

    plt.show()


if __name__ == '__main__':
    main()
