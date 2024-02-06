import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 15})

def normalize_list(lst):
    least = min(lst)
    largest = max(lst)
    alpha = largest - least
    out = []

    for n in lst:
        if n < 0:
            out.append(0)
        else:
            out.append((n - least) / alpha)

    return out

def main():
    data_df = pd.read_csv("concrete.csv")

    # train, test = train_test_split(data_df, test_size=0.30)

    train_X = data_df.drop("strength", axis=1)
    train_y = np.array(data_df["strength"])

    standardizer = StandardScaler()
    train_X = standardizer.fit_transform(train_X)

    # test_X = test.drop("strength", axis=1)
    # test_y = np.array(test["strength"])

    hidden_layers = [18, 36, 72, 144, 288, 576, 1152, 576, 288, 144, 72, 36, 18, 9, 3]
    mlpreg = MLPRegressor(hidden_layer_sizes=hidden_layers, activation="relu", n_iter_no_change=7000,
                          solver="adam", early_stopping=True, max_iter=600, verbose=True)
    mlpreg.fit(train_X, train_y)

    loss_curve = normalize_list(mlpreg.loss_curve_[30:])
    validation_scores = normalize_list([1 - i for i in mlpreg.validation_scores_[30:]])
    iteration = [i for i in range(len(loss_curve))]

    plt.plot(iteration, loss_curve, "r", label="loss")
    plt.plot(iteration, validation_scores, "b", label="validation")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
