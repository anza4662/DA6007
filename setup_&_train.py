import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

import networks_width

import pickle as pk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Some code taken from https://machinelearningmastery.com/building-a-regression-model-in-pytorch/

device = torch.device("cpu")


def plot_2D_results_netsize(train, test, model_names, title):
    nets = [i for i in range(1, len(model_names) + 1)]
    epochs = [i for i in range(1, len(train[0]) + 1)]

    v_min = 0
    v_max = 15

    E, N = np.meshgrid(epochs, nets)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
    fig.tight_layout(pad=5)

    contf1_ = ax[0].contourf(E, N, train, levels=400, vmin=v_min, vmax=v_max)
    ax[0].set_title("Training Loss")
    ax[0].set_yticks(nets, model_names)
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Network")

    contf2_ = ax[1].contourf(E, N, test, levels=400, vmin=v_min, vmax=v_max)
    ax[1].set_title("Test Loss")
    ax[1].set_yticks(nets, model_names)
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Network")

    fig.colorbar(contf1_, ax=ax[0])
    fig.colorbar(contf2_, ax=ax[1])

    fig.suptitle("Train and test errors for increasing network sizes.\n" + title)
    plt.subplots_adjust(top=0.80)
    plt.show()


def network_to_file(history):
    filename = "hiya.txt"
    with open(filename, "w") as file:
        pk.dump(history, file)
        print(f"Wrote history to: " + filename + ".")


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
    X = np.array(data.drop("val", axis=1))
    y = np.array(data["val"])

    if data_set_size != 500000:
        X = X[0:data_set_size]
        y = y[0:data_set_size]

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, shuffle=True)

    train_y = add_noise(train_y, delta_noise)
    test_y = add_noise(test_y, delta_noise)

    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)

    train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
    train_y = torch.tensor(train_y, dtype=torch.float32).reshape(-1, 1).to(device)

    test_X = torch.tensor(test_X, dtype=torch.float32).to(device)
    test_y = torch.tensor(test_y, dtype=torch.float32).reshape(-1, 1).to(device)

    return train_X, test_X, train_y, test_y


def train_one(model, data_set, data_set_size, delta_noise,
              minibatch_size, learning_rate, betas_adam,
              n_epochs, plot_history):
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
        "x1": train_x[:, 0].clone().tolist(),
        "x2": train_x[:, 1].clone().tolist(),
        "title": ""
    }

    weight_cut = int(n_epochs / 9)

    batch_start = torch.arange(0, len(train_x), minibatch_size)
    batches = []
    for start in batch_start:
        x_batch = train_x[start:start + minibatch_size].to(device)
        y_batch = train_y[start:start + minibatch_size].to(device)
        batches.append((x_batch, y_batch))

    with tqdm.tqdm(range(1, n_epochs + 1), ncols=150, colour="green") as bar:
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
                history["diff_func_per_epoch"].append(diff)

            bar.set_postfix({"Train": loss.item(), "Test": val_loss.item()})

    if plot_history:
        history["title"] = (f"Network training stats. Architecture = {type(model).__name__}, "
                            f"epochs = {n_epochs}, batch size = {minibatch_size}, \n delta_noise = {delta_noise}, "
                            f"data set size = {data_set_size / 1000}k, learning rate = {learning_rate}, "
                            f"betas = {betas_adam}")

        network_to_file(history)

    return history


def train_multiple(data_set, data_set_size, delta_noise,
                   minibatch_size, learning_rate, betas_adam,
                   n_epochs):
    # models = [networks.NetD1(), networks.NetD2(), networks.NetD3(), networks.NetD4(),
    #          networks.NetD5(), networks.NetD6(), networks.NetD7()]

    models = [networks_width.NetW1(), networks_width.NetW2(), networks_width.NetW3(), networks_width.NetW4(),
              networks_width.NetW5(), networks_width.NetW6(), networks_width.NetW7(), networks_width.NetW8(),
              networks_width.NetW9(), networks_width.NetW10(), networks_width.NetW11(), networks_width.NetW12(),
              networks_width.NetW13(), networks_width.NetW14(), networks_width.NetW15(), networks_width.NetW16()]

    results_train = []
    results_test = []
    model_names = []

    for model in models:
        model.to(device)
        history = train_one(model, data_set, data_set_size, delta_noise,
                            minibatch_size, learning_rate, betas_adam,
                            n_epochs, False)
        results_train.append(history["train_loss"])
        results_test.append(history["val_loss"])

        model_names.append(type(model).__name__)

    title = (f"Network parameters: epochs = {n_epochs}, batch size = {minibatch_size}, delta_noise = {delta_noise}, \n"
             f"data set size = {data_set_size / 1000}k, learning rate = {learning_rate}, betas = {betas_adam}")

    plot_2D_results_netsize(results_train, results_test, model_names, title)


def test_data(model, data_set_size, delta_noise, device, minibatch_size, learning_rate, betas_adam, n_epochs):
    results_train = []
    results_test = []
    k_lst = [2.01, 2.408, 2.806, 3.204, 3.602, 4.0]

    for data_k in k_lst:
        data_set = f"data/data_set/data5var_k={data_k}_20k"
        print("Training on dataset: ", data_set)
        history = train_one(model, data_set, data_set_size, delta_noise,
                            minibatch_size, learning_rate, betas_adam,
                            n_epochs, False)
        results_train.append(history["train_loss"])
        results_test.append(history["val_loss"])

    title = (f"Network parameters: Architecture = {type(model).__name__},  epochs = {n_epochs}, "
             f"batch size = {minibatch_size}, \n delta_noise = {delta_noise}, data set size = {data_set_size / 1000}k, "
             f"learning rate = {learning_rate}, betas = {betas_adam}")


def main():
    # torch.manual_seed(0)
    # np.random.seed(0)

    n_epochs = 27
    minibatch_size = 10

    # Data set settings
    delta_noise = 1
    data_set = "data/data_set/data5var_k=3.204_20k"
    data_set_size = 2000

    # Adam parameters
    learning_rate = 1e-4
    betas_adam = (0.9, 0.999)
    # model = networks.NetD3().to(device)
    model = networks_width.NetW1().to(device)

    # test_data(model, data_set_size, delta_noise, device,
    #          minibatch_size, learning_rate, betas_adam, n_epochs)

    his = train_one(model, data_set, data_set_size, delta_noise,
                    minibatch_size, learning_rate, betas_adam,
                    n_epochs, True)

    # train_multiple(data_set, data_set_size, delta_noise,
    #               device, minibatch_size, learning_rate, betas_adam,
    #               n_epochs)


if __name__ == '__main__':
    main()
