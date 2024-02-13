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
    df = create_data()
    visualize_data(df)


def visualize_data(df):
    print(df.describe().transpose())
    sns.pairplot(df, kind="hist", diag_kind="kde")
    df.to_csv("data_5_features_100k")
    plt.show()


def create_data():
    n = 100000
    a, b = -2, 2
    data = []
    for i in range(n):
        x1 = random.uniform(a, b)
        x2 = random.uniform(a, b)
        x3 = random.uniform(a, b)
        x4 = random.uniform(a, b)
        x5 = random.uniform(a, b)

        val = 0.5 * math.sin(x1*x2*x3*x4*x5) - 5 * math.atan(x3*x4) + (1/(1 + (x2 + (x4*x5))**2)) + 0.01 * (((x1-x3)**5) - 10 * ((x2 * x5)**2))

        row = [x1, x2, x3, x4, x5, val]
        data.append(row)

    df = pd.DataFrame(data, columns=["x1", "x2", "x3", "x4", "x5", "val"])
    df.to_csv("data_5_features_100k", index=False)
    return df


if __name__ == '__main__':
    main()
