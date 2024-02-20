import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.layout_engine as le


# Some code taken from https://machinelearningmastery.com/building-a-regression-model-in-pytorch/

# Data (Parameter for non-linearity)                    DONE
# Noise variance                                        DONE
# Architecture                                          DONE
# Batch normalization                                   DONE
# Skip connections                                      DONE
# Initialization (How is the network initialized)       DONE
# Weight distribution                                   DONE
# Adam parameters
# Why does adam have 18 different states?
# Should batch norm. have learnable parameters? And why before ReLu?
# How much noise do we want?
# Skip connection to output?
# Variance for initialization?

#                      x
#             ------   |
#             |    linear(5, 10)
#             |    batch norm
#             |       relu
#             |    linear(10,20)
#             |    batch norm
#             -----> + |
#                     relu
#             ------   |
#             |    linear(20,10)
#             |    batch norm
#             |       relu
#             |    linear(10,5)
#             |    batch norm
#             -----> + |
#                     relu
#             ------   |
#             |    linear(5,2)
#             |    batch norm
#             |       relu
#             |    linear(2,1)
#             -----> + |
#                      y

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)

        self.lin3 = nn.Linear(20, 10)
        self.lin4 = nn.Linear(10, 5)

        self.lin5 = nn.Linear(5, 2)
        self.lin6 = nn.Linear(2, 1)

        batch_learnable_params = False
        self.bn1 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn2 = nn.BatchNorm1d(20, affine=batch_learnable_params)
        self.bn3 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn4 = nn.BatchNorm1d(5, affine=batch_learnable_params)
        self.bn5 = nn.BatchNorm1d(2, affine=batch_learnable_params)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 5)
        self.skip3 = nn.Linear(5, 1)

    def forward(self, x):
        z1 = self.lin1(x)
        z_skip1 = self.skip1(x)
        z2 = torch.relu(self.bn1(z1))
        z3 = self.lin2(z2)
        z4 = torch.relu(self.bn2(z3) + z_skip1)

        z_skip2 = self.skip2(z4)
        z5 = self.lin3(z4)
        z6 = torch.relu(self.bn3(z5))
        z7 = self.lin4(z6)
        z8 = torch.relu(self.bn4(z7) + z_skip2)

        z_skip3 = self.skip3(z8)
        z9 = self.lin5(z8)
        z10 = torch.relu(self.bn5(z9))
        z11 = self.lin6(z10) + z_skip3
        return z11


def get_weights(model):
    layer_weights = {}
    layers = [("lay1", model.lin1), ("lay2", model.lin2), ("lay3", model.lin3),
              ("lay4", model.lin4), ("lay5", model.lin5), ("lay6", model.lin6)]

    for name, lay in layers:
        layer_weights[name] = []
        for w in lay.weight.tolist():
            for k in w:
                layer_weights[name].append(k)

    return layer_weights


def init_normal(module):
    if type(module) == nn.Linear:
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

    # Cpu is faster for smaller networks (size < 100)
    # dev = "cuda" or dev = "cpu"
    dev = "cpu"
    n_epochs = 240
    minibatch_size = 10
    delta_noise = 1
    data_set = "data/data5features_0to20_50k"
    device = torch.device(dev)
    learning_rate = 1e-2
    betas_adam = (0.87, 0.997)
    data_set_size = 10000

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

    network_architecture = [5, 10, 20, 10, 5, 2, 1]

    model = Net().to(device)
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

    weight_cut = int(n_epochs / 6)

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
    fig, axs = plt.subplots(3, 3, figsize=(24, 16))
    fig.set_layout_engine()
    fig.suptitle(f"Network training stats. Architecture = {network_architecture}  epochs = {n_epochs}, "
                 f"batch size = {minibatch_size}, \n delta_noise = {delta_noise}, data set size = {data_set_size /1000}k, "
                 f"learning rate = {learning_rate}, betas = {betas_adam}")
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

    alpha_lst = [0.9, 0.5, 0.4, 0.7, 0.8, 0.9]
    for (subplt, dct, ep) in zip(flattened_axs, weights_per_layer_epoch, epoch_saved):

        for (name, a) in zip(dct.keys(), alpha_lst):
            subplt.hist(dct[name], histtype="bar", bins=25, label=name, alpha=a)
        subplt.set_title(f"Epoch {ep}")
        subplt.legend()

    fig.text(0.5, 0.02, "Weight value", ha="center", va="center")
    fig.text(0.02, 0.3, "Number of occurrences", ha="center", va="center", rotation="vertical")

    plt.show()


if __name__ == '__main__':
    main()
