import random
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 8 * np.sin(0.1 * x) + 6 * np.cos(0.5 * x) - 4 * np.sin(2 * x) + 2 * np.cos(x) - np.cos(0.2 * x)


def functionPlot():
    domain = [random.uniform(-5, 5) for i in range(1000)]
    domain.sort()
    image = [f(x) for x in domain]
    n = 0.9

    v = (np.var(image) / np.abs(np.max(image) - np.min(image))) * n
    noise = np.random.normal(0, v, len(image))
    image += noise

    plt.plot(domain, image, color='red')
    plt.title(f"noise with var = var(f) / range(f) \n with factor {n}")
    plt.xlabel("x")
    plt.ylabel("f(x) + noise")
    plt.show()


def visualize_data(df):
    print(df.describe().transpose())
    sns.pairplot(df, kind="hist", diag_kind="kde")
    df.to_csv("data_5_features_100k")
    plt.show()


# If we want val to be in range 0 to n we take a to be zero and b to be the fifth root of n.

def create_data():
    n = 50000
    a, b = -2, 2
    k = 5.2
    c = 4
    data = []

    for i in range(n):
        x1 = random.uniform(a, b)
        x2 = random.uniform(a, b)
        x3 = random.uniform(a, b)
        x4 = random.uniform(a, b)
        x5 = random.uniform(a, b)

        prod1 = x1
        prod2 = x3 * x4
        prod3 = x2 * x5

        val = k * (np.sin((c - k) * prod1) + np.cos((c - k) * prod2) - np.sin((c - k) * prod3))

        row = [x1, x2, x3, x4, x5, val]
        data.append(row)

    df = pd.DataFrame(data, columns=["x1", "x2", "x3", "x4", "x5", "val"])
    df.to_csv(f"data5var_k={k}_50k", index=False)
    print(df.describe().transpose())


def create_data_V3():
    n = 20000
    k = 2.806
    var = 1
    c = 2
    data = []
    cov = [[var,0,0,0,0], [0,var,0,0,0], [0,0,var,0,0], [0,0,0,var,0], [0,0,0,0,var]]

    for i in range(n):
        x = np.random.multivariate_normal(np.zeros(5), cov, 1)[0]

        prod1 = x[0]
        prod2 = x[2] * x[3]
        prod3 = x[1] * x[4]

        val = k * (np.sin((c - k) * prod1) + np.cos((c - k) * prod2) - np.sin((c - k) * prod3))
        row = np.append(x, val)
        data.append(row)

    df = pd.DataFrame(data, columns=["x1", "x2", "x3", "x4", "x5", "val"])
    df.to_csv(f"data5var_k={k}_20k_normal", index=False)
    print(df.describe().transpose())


def create_data_multiple_k():
    n = 10000
    a, b = -2, 2

    k_range = np.linspace(2.01, 4, 6)
    k_range = [np.round(k, 4) for k in k_range]
    print(k_range)

    for k in k_range:
        data = []
        for i in range(n):
            x1 = random.uniform(a, b)
            x2 = random.uniform(a, b)
            x3 = random.uniform(a, b)
            x4 = random.uniform(a, b)
            x5 = random.uniform(a, b)

            prod1 = x1
            prod2 = x3 * x4
            prod3 = x2 * x5

            val = k * (np.sin((2 - k) * prod1) + np.cos((2 - k) * prod2) - np.sin((2 - k) * prod3))

            row = [x1, x2, x3, x4, x5, val]
            data.append(row)

        df = pd.DataFrame(data, columns=["x1", "x2", "x3", "x4", "x5", "val"])
        df.to_csv(f"data5var_k={k}_20k", index=False)
        print(df.describe().transpose())


def main():
    create_data_V3()


if __name__ == '__main__':
    main()
