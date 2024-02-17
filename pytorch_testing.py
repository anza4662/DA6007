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

# Data (Parameter for non-linearity)
# Noise variance
# Architecture
# Batch normalization (why before relu?)
# Skip connections (Every two layers)
# Initialization (How is the network initialized)

# Some code taken from https://machinelearningmastery.com/building-a-regression-model-in-pytorch/


def add_noise(arr):
    v = np.var(arr) / np.abs(np.max(arr) - np.min(arr))
    noise = np.random.normal(0, v, arr.shape)
    arr += noise
    return arr


def main():
    # Cpu is faster for smaller networks (size < 100)
    # dev = "cuda" or dev = "cpu"
    dev = "cuda"
    n_epochs = 10
    minibatch_size = 10

    device = torch.device(dev)

    data = pd.read_csv("data_5_features_20k", index_col=0)
    X = np.array(data.drop("val", axis=1))
    y = np.array(data["val"])

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, shuffle=True)

    train_y = add_noise(train_y)

    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).reshape(-1, 1).to(device)

    test_X = torch.tensor(test_X, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y, dtype=torch.float32).reshape(-1, 1).to(device)

    network_architecture = [5, 10, 20, 10, 5, 2, 1]

    model = nn.Sequential(
        nn.Linear(5, 10),
        nn.ReLU(),
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
        nn.ReLU(),
        nn.Linear(2, 1),
    ).to(device)

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

                y_pred = model(X_batch)

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
    fig.suptitle(f"Network training stats. Architecture = {network_architecture} \n epochs = {n_epochs}, batch size = {minibatch_size}")

    # Plot history
    axs[0].plot(history["val_loss"], label="val_loss")
    axs[0].plot(history["train_loss"], label="train_loss")
    axs[0].set_ylabel("MSE")
    axs[0].set_xlabel("Epoch")
    axs[0].legend()

    axs[1].plot(history["grad_norm"], color="red")
    axs[1].set_ylabel("Gradient norm (L2)")
    axs[1].set_xlabel("Epoch")

    params_lst = []
    for i in model.parameters():
        for j in i.tolist():
            if type(j) is not float:
                for k in j:
                    params_lst.append(k)
            else:
                params_lst.append(j)

    # Plot weight distribution
    axs[2].hist(params_lst, bins=20)
    axs[2].set_ylabel("Number of occurrences")
    axs[2].set_xlabel("Parameter values")

    plt.show()


if __name__ == '__main__':
    main()
