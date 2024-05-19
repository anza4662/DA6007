import sys
import threading
from threading import Thread
import time
from datetime import datetime

import torch
import torch.utils.data as data_util
import torch.nn as nn
import torch.optim as optim

import pickle as pk
import pandas as pd
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import networks

device = torch.device("cpu")


def data_to_file(history):
    name = datetime.now().strftime("%d%m%Y_%H%M%S")

    filename = f"produced_files/emc_data_{name}.txt"
    with open(filename, "wb") as file:
        pk.dump(history, file)
        print(f"Wrote history to: " + filename + ".")


def init_normal(module):
    if type(module) is nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)


def f(x_n, k, c):
    prod1 = x_n[0] * x_n[2]
    prod2 = x_n[1] * x_n[0] * x_n[3]
    prod3 = x_n[2] * x_n[1] * x_n[4]

    return k * (np.sin((c - k) * prod1) + np.cos((c - k) * prod2) - np.sin((c - k) * prod3))


def get_data(sample_size, noise_std, k, c, x_range, three_var):
    x = []
    y = []

    for i in range(sample_size):
        if three_var:
            x_n = np.array([np.random.uniform(-x_range, x_range), np.random.uniform(-x_range, x_range),
                            np.random.uniform(-x_range, x_range), 1, 1])
        else:
            x_n = np.array([np.random.uniform(-x_range, x_range), np.random.uniform(-x_range, x_range),
                            np.random.uniform(-x_range, x_range), np.random.uniform(-x_range, x_range),
                            np.random.uniform(-x_range, x_range)])

        val = f(x_n, k, c)

        x.append(x_n)
        y.append(val)

    x = np.array(x)
    y = np.array(y)

    y += np.random.normal(0, noise_std, y.shape)

    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)

    return x, y


def calc_std_noise_3_var(k, c, x_range):
    print("Calculating standard deviation...")
    n = 136
    x = np.linspace(-x_range, x_range, n)

    vals = []

    for i in range(n):
        for j in range(n):
            for s in range(n):
                x_n = [x[i], x[j], x[s], 1, 1]
                vals.append(f(x_n, c, k))

    return np.std(vals) / 10


def the_big_emc_test(sample_size, k, c, x_range, noise_std):
    three_var = True
    minibatch_size = 20
    learning_rate = 0.1
    betas_adam = (0.9, 0.999)

    # architecture = [6, 12, 7]
    architecture = [6, 12, 18, 15, 12, 9, 7]
    # architecture = [6, 9, 13, 18, 27, 23, 17, 15, 12, 10, 7]
    n_epochs = 200

    loss_fn = nn.MSELoss()
    batch_start = torch.arange(0, sample_size, minibatch_size)
    print(batch_start)
    test_models = {}

    for i in range(10):
        train_x, train_y = get_data(sample_size, noise_std, k, c, x_range, three_var)

        for j in range(5):
            model = networks.NetMedium(architecture).to(device)
            # model = networks.NetSmall(architecture).to(device)
            model.apply(init_normal)

            if i == 0:
                n_params = sum(p.numel() for p in model.parameters())
                setup = (f" Number of parameters: {n_params} \n "
                         f"Minibatch size: {minibatch_size} \n "
                         f"Adam learning rate: {learning_rate} \n "
                         f"Adam betas: {betas_adam} \n "
                         f"k: {k} \n "
                         f"c: {c} \n "
                         f"Noise std: {noise_std} \n "
                         f"Architecture: {architecture} \n "
                         f"Sample size: {sample_size} \n"
                         )

                print(setup)

            optimizer = optim.Adam(model.parameters(), weight_decay=0, lr=learning_rate, betas=betas_adam)

            batches = []
            for start in batch_start:
                x_batch = train_x[start:start + minibatch_size].to(device)
                y_batch = train_y[start:start + minibatch_size].to(device)
                batches.append((x_batch, y_batch))

            test_models[str(i) + str(j)] = {
                "model": model,
                "optimizer": optimizer,
                "batches": batches,
                "train_x": train_x,
                "train_y": train_y,
            }

    n_batches = len(batches)
    n_epochs_list = range(1, n_epochs + 1)

    var_train_error = []
    y_ci = []
    avg_train_loss_per_epoch = []
    avg_first_moms_per_epoch = []
    avg_second_moms_per_epoch = []

    with tqdm.tqdm(n_epochs_list, ncols=100) as bar:

        for epoch in bar:

            train_loss_this_epoch = []
            var_train_error_per_epoch = []
            first_mom = []
            second_mom = []

            for test_model in test_models:

                avg_loss_per_epoch_per_model = 0
                model = test_models[test_model]["model"]
                optimizer = test_models[test_model]["optimizer"]
                batches = test_models[test_model]["batches"]

                model.train()

                for (batch_x, batch_y) in batches:
                    pred_y = model.forward(batch_x)
                    loss = loss_fn(pred_y, batch_y)
                    avg_loss_per_epoch_per_model += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                first_mom.append(np.average(
                    [float(torch.norm(val["exp_avg"])) for val in optimizer.state_dict()["state"].values()]
                ))

                second_mom.append(np.average(
                    [float(torch.norm(val["exp_avg_sq"])) for val in optimizer.state_dict()["state"].values()]
                ))

                model.eval()

                pred = model.forward(test_models[test_model]["train_x"])
                diff = test_models[test_model]["train_y"] - pred
                var_train_error_per_epoch.append(np.var(diff.tolist()))

                avg_loss_per_epoch_per_model = avg_loss_per_epoch_per_model / n_batches

                train_loss_this_epoch.append(avg_loss_per_epoch_per_model)

            var_train_error.append(np.mean(var_train_error_per_epoch))

            avg_train_loss_this_epoch = np.mean(train_loss_this_epoch)
            std_train_loss_this_epoch = np.std(train_loss_this_epoch)

            avg_train_loss_per_epoch.append(avg_train_loss_this_epoch)

            avg_first_moms_per_epoch.append(np.mean(first_mom))
            avg_second_moms_per_epoch.append(np.mean(second_mom))

            z_score = 3.291  # For 99.9% confidence interval
            ci_lower = avg_train_loss_this_epoch - (
                        z_score * std_train_loss_this_epoch / np.sqrt(len(train_loss_this_epoch)))
            y_ci.append(ci_lower)

            bar.set_postfix({"Train": avg_train_loss_this_epoch, "ci_l": ci_lower})

    data_to_file({"y_ci": y_ci,
                  "mean_loss": avg_train_loss_per_epoch,
                  "sample_size": sample_size,
                  "var_train_error": var_train_error,
                  "setup": setup,
                  "std_noise": noise_std,
                  "first_mom": avg_first_moms_per_epoch,
                  "second_mom": avg_second_moms_per_epoch
                  })


def main():
    k = 2.05
    c = 2
    x_range = 3.5
    # noise_std = 0.0083
    noise_std = calc_std_noise_3_var(k, c, x_range)
    # print(calc_std_noise_3_var(k, c, x_range))

    data_sizes = range(20, 301, 20)

    for i in data_sizes:
         the_big_emc_test(i, k, c, x_range, noise_std)


if __name__ == '__main__':
    main()
