import random
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import pickle as pk

def f(x_n, c, k):
    prod1 = x_n[0] * x_n[2]
    prod2 = x_n[1] * x_n[0] * x_n[3]
    prod3 = x_n[2] * x_n[1] * x_n[4]
    return k * (np.sin((c - k) * prod1) + np.cos((c - k) * prod2) - np.sin((c - k) * prod3))


def create_data_set():
    n = 50000
    k = 3.8
    c = 3
    x_range = 3
    std_noise = 0.447
    three_var = False

    train_x = []
    train_y = []
    test_x = []
    test_y = []

    test_size = n

    for i in range(n):
        if three_var:
            x_n_train = np.array([np.random.uniform(-x_range, x_range), np.random.uniform(-x_range, x_range),
                                  np.random.uniform(-x_range, x_range), 1, 1])
            train_val = f(x_n_train, k, c)

            x_n_test = np.array([np.random.uniform(-x_range, x_range), np.random.uniform(-x_range, x_range),
                                 np.random.uniform(-x_range, x_range), 1, 1])

            test_val = f(x_n_test, k, c)

        else:
            x_n_train = np.array([np.random.uniform(-x_range, x_range), np.random.uniform(-x_range, x_range),
                                  np.random.uniform(-x_range, x_range), np.random.uniform(-x_range, x_range),
                                  np.random.uniform(-x_range, x_range)])
            train_val = f(x_n_train, k, c)

            x_n_test = np.array([np.random.uniform(-x_range, x_range), np.random.uniform(-x_range, x_range),
                                 np.random.uniform(-x_range, x_range), np.random.uniform(-x_range, x_range),
                                 np.random.uniform(-x_range, x_range)])

            test_val = f(x_n_test, k, c)

        train_x.append(x_n_train)
        train_y.append(train_val)

        if i <= test_size:
            test_x.append(x_n_test)
            test_y.append(test_val)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    # Add noise
    train_y += np.random.normal(0, std_noise, train_y.shape)
    test_y += np.random.normal(0, std_noise, test_y.shape)

    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    test_x = torch.tensor(test_x, dtype=torch.float32)
    test_y = torch.tensor(test_y, dtype=torch.float32).reshape(-1, 1)

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32).reshape(-1, 1)

    data = {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "n": n,
        "std_noise": std_noise,
        "x_range": x_range,
        "c": c,
        "k": k
    }

    filename = f"data/data_five_var_v8.txt"

    with open(filename, "wb") as file:
        pk.dump(data, file)
        print(f"Wrote history to {filename}.")


def calc_std_noise_3_var(k, c, x_range):
    n = 200
    x = np.linspace(-x_range, x_range, n)

    vals = []

    for i in range(n):
        for j in range(n):
            for s in range(n):
                x_n = [x[i], x[j], x[s], 1, 1]
                vals.append(f(x_n, c, k))

    return np.std(vals)


def calc_std_noise_5_var(k, c, x_range):
    n = 30
    x = np.linspace(-x_range, x_range, n)

    vals = []

    for i in range(n):
        for j in range(n):
            for d in range(n):
                for t in range(n):
                    for s in range(n):
                        x_n = [x[i], x[j], x[d], x[t], x[s]]
                        vals.append(f(x_n, c, k))

    return np.std(vals)


def three_d_plot_func(k, c, x_range):
    n = 50

    fig = plt.figure(figsize=(24, 15))
    x = np.linspace(-x_range, x_range, n)

    vals1 = []
    vals2 = []

    for i in range(n):
        for j in range(n):
            x_n1 = [x[i], 1, x[j], 1, 1]
            x_n2 = [x[i], x[j], 1, 1, 1]

            vals1.append(f(x_n1, c, k))
            vals2.append(f(x_n2, c, k))

    vals1 = np.array(vals1)
    vals2 = np.array(vals2)

    vals1 = vals1.reshape(n, n)
    vals2 = vals2.reshape(n, n)

    xi1, xi2 = np.meshgrid(x, x)

    subp = fig.add_subplot(1, 2, 1, projection="3d")
    subp.plot_wireframe(xi1, xi2, vals1, color="green")
    subp.set_title("x1, x3")

    subp = fig.add_subplot(1, 2, 2, projection="3d")
    subp.plot_wireframe(xi1, xi2, vals2, color="green")
    subp.set_title("x1, x2")
    plt.show()


def show_func():
    k = 2.5
    c = 2
    n = 500
    x_range = 1.5

    x = []
    y = []

    for i in range(n):
        x_n = np.array([np.random.uniform(-x_range, x_range), np.random.uniform(-x_range, x_range),
                         np.random.uniform(-x_range, x_range), np.random.uniform(-x_range, x_range),
                         np.random.uniform(-x_range, x_range)])
        val = f(x_n, c, k)

        x.append(x_n)
        y.append(val)

    x = np.array(x)
    y = np.array(y)

    data = pd.DataFrame(
        {"x1": [i[0] for i in x],
         "x2": [i[1] for i in x],
         "x3": [i[2] for i in x],
         "x4": [i[3] for i in x],
         "x5": [i[4] for i in x],
         "y": y
         }
    )

    gr = sns.pairplot(data, vars=["x1", "x2", "x3", "x4", "x5", "y"], kind="scatter")
    gr.map(plt.scatter, alpha=0.8)
    gr.add_legend()
    plt.show()


def main():
    k = 3.8
    c = 3
    x_range = 3

    # print("Recommended noise std. =", calc_std_noise_5_var(k, c, x_range))
    # print("Recommended noise std. =", calc_std_noise_3_var(k, c, x_range))

    # three_d_plot_func(k, c, x_range)

    create_data_set()

    # show_func()
    # lst = np.linspace(2.1, 6, 10)
    #
    # for k in lst:
    #     functionPlot(k)


if __name__ == '__main__':
    main()
