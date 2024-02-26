import random
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import datashader as ds, colorcet


def main():
    # df = create_data()
    # visualize_data(df)
    functionPlot()


def f(x):
    return 8 * np.sin(0.1 * x) + 6 * np.cos(0.5 * x) - 4 * np.sin(2 * x) + 2 * np.cos(x) - np.cos(0.2 * x)


def functionPlot():
    domain = [random.uniform(-100, 100) for i in range(1000)]
    domain.sort()
    image = [f(x) for x in domain]
    n = 0

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
    a, b = 0, np.power(4, 1 / 3)
    data = []
    for i in range(n):
        x1 = random.uniform(a, b)
        x2 = random.uniform(a, b)
        x3 = random.uniform(a, b)
        x4 = random.uniform(a, b)
        x5 = random.uniform(a, b)

        prod1 = x1 * x2
        prod2 = x3 * x4 * x5
        prod3 = x2 * x5
        prod4 = x1 * x3

        # f1
        # val = np.sin(1.9 * prod1) + 2 * np.cos(0.2 * prod3) - 2 * np.sin(1.6 * prod2) - np.cos(2 * prod4)

        # f2
        val = 8 * np.sin(0.1 * prod1) + 6 * np.cos(0.5 * prod2) - 4 * np.sin(2 * prod4) + 2 * np.cos(prod3) - np.cos(
            0.2 * prod1)
        print(val)
        row = [x1, x2, x3, x4, x5, val]
        data.append(row)

    df = pd.DataFrame(data, columns=["x1", "x2", "x3", "x4", "x5", "val"])
    df.to_csv("data5features_+-4_50k_v2", index=False)
    print(df.describe().transpose())
    return df


if __name__ == '__main__':
    main()
