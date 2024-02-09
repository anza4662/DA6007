import random
import math
import pandas


def main():
    n = 100000
    data = []
    for i in range(n):
        x1, x2, x3, x4, x5 = random.random(), random.random(), random.random(), random.random(), random.random()
        val = math.sin(x1*x2*x3*x4*x5) * (math.e ** (x1*x3*x5)) + (2.3 * math.e ** (x2*x4*x5)) - math.atan(x1*x2*x3)
        row = [x1, x2, x3, x4, x5, val]
        data.append(row)

    df = pandas.DataFrame(data, columns=["x1", "x2", "x3", "x4", "x5", "val"])
    print(df.describe().transform)
    df.to_csv("data_5_features")


if __name__ == '__main__':
    main()
