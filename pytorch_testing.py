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

# Data (Parameter for non-linearity)                    DONE
# Noise variance                                        DONE
# Architecture                                          DONE
# Batch normalization (why before relu?)                DONE
# Skip connections (Every two layers)                   DONE
# Initialization (How is the network initialized)       DONE

# Some code taken from https://machinelearningmastery.com/building-a-regression-model-in-pytorch/


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
    # Cpu is faster for smaller networks (size < 100)
    # dev = "cuda" or dev = "cpu"
    dev = "cuda"
    n_epochs = 300
    minibatch_size = 10
    delta_noise = 0.5
    data_set = "data/data5features_0to20_50k"
    device = torch.device(dev)

    data = pd.read_csv(data_set)

    X = np.array(data.drop("val", axis=1))
    y = np.array(data["val"])

    # X = X[0:20000]
    # y = y[0:20000]

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
    optimizer = optim.Adam(model.parameters(), weight_decay=0)

    batch_start = torch.arange(0, len(train_X), minibatch_size)

    best_mse = np.inf
    best_weights = None
    history = {
        "val_loss": [],
        "train_loss": [],
        "grad_norm": []
    }

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

        history["train_loss"].append(loss.item())
        history["grad_norm"].append(grad_norm)

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

    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
    fig.set_layout_engine(le.ConstrainedLayoutEngine(wspace=0.05, w_pad=0.1, h_pad=0.1))
    fig.suptitle(f"Network training stats. Architecture = {network_architecture} \n epochs = {n_epochs}, "
                 f"batch size = {minibatch_size}, \n delta_noise = {delta_noise}, data set = {data_set}")

    # Plot history
    axs[0].plot(history["val_loss"], label="val_loss")
    axs[0].plot(history["train_loss"], label="train_loss")
    axs[0].set_ylabel("MSE")
    axs[0].set_xlabel("Epoch")
    axs[0].legend()

    axs[1].plot(history["grad_norm"], color="red")
    axs[1].set_ylabel("Gradient norm (L2)")
    axs[1].set_xlabel("Epoch")

    layer_weights = {}

    layers = [("lay1", model.lin1), ("lay2", model.lin2), ("lay3", model.lin3),
              ("lay4", model.lin4), ("lay5", model.lin5), ("lay6", model.lin6)]

    for name, lay in layers:
        layer_weights[name] = []
        for w in lay.weight.tolist():
            for k in w:
                layer_weights[name].append(k)

    # Plot weight distribution per layer
    for name in layer_weights.keys():
        axs[2].hist(layer_weights[name], bins=25, alpha=0.5, label=name)
    axs[2].legend()
    axs[2].set_ylabel("Number of occurrences")
    axs[2].set_xlabel("Parameter values")

    plt.show()


if __name__ == '__main__':
    main()
