import random
import math
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    n = 100000
    a, b = -0.5, 1
    data = []
    for i in range(n):
        x1 = random.uniform(a, b)
        x2 = random.uniform(a, b)
        x3 = random.uniform(a, b)
        x4 = random.uniform(a, b)
        x5 = random.uniform(a, b)
        # val = math.sin(x1*x2*x3*x4*x5) + math.atan(x1*x2) - math.e ** (x4*x5) - (1/(1 + (x1*x2)**2)) + (x3**5)
        val = 0.2 * math.atan(x4*x3) * (1/(1 + x2**2)) - (x3**5) + (2 * math.e ** (x3*x5) + ((x4**5) - 2 * (x1**3))) + (0.05 * math.e ** (x4*x5)) + math.sin(x1*x2 - x3*x4 + x5)/(7+x2)
        row = [x1, x2, x3, x4, x5, val]
        data.append(row)

    df = pandas.DataFrame(data, columns=["x1", "x2", "x3", "x4", "x5", "val"])
    print(df.describe().transform)
    sns.pairplot(df, kind="hist", diag_kind="kde")
    df.to_csv("data_5_features")
    plt.show()

if __name__ == '__main__':
    main()
