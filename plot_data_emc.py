import matplotlib.pyplot as plt
import numpy as np
import pickle as pk


def get_data_from_file(filename):
    with open(filename, "rb") as file:
        history = pk.load(file)
    return history


def plot_emc_per_epoch():
    files = [
        "emc_data_01052024_131045.txt",
        "emc_data_01052024_131158.txt",
        "emc_data_01052024_131337.txt",
        "emc_data_01052024_131540.txt",
        "emc_data_01052024_131801.txt",
        "emc_data_01052024_132051.txt",
        "emc_data_01052024_132400.txt",
        "emc_data_01052024_132731.txt",
        "emc_data_01052024_133128.txt",
        "emc_data_01052024_133547.txt",
        "emc_data_01052024_134040.txt",
        "emc_data_01052024_134626.txt",
        "emc_data_01052024_135238.txt",
        "emc_data_01052024_135920.txt",
        "emc_data_01052024_141120.txt"
    ]

    fig = plt.figure(figsize=(9, 6))
    fig.tight_layout(pad=2)

    data = []

    for file in files:
        history = get_data_from_file("/home/anza/kanidiatarbete/emc_estimation/final test/data/" + file)
        std_noise = history["std_noise"]
        data.append(zip(history["y_ci"], [history["sample_size"] for i in range(1, 120)]))

    zipped = zip(*data)
    emc = []

    for epoch in zipped:
        lst = [(x, y) for (x, y) in epoch if x < std_noise]

        if len(lst) > 0:
            max_val = max(lst, key=lambda x: x[1])
            emc.append(max_val[1])
        else:
            emc.append(None)

    epochs = [x for x in range(1, len(emc) + 1)]

    # p = np.poly1d(np.polyfit(epochs, emc1, deg=5))
    # emc1_fitted = p(epochs)

    axs = fig.add_subplot(1, 1, 1)
    axs.set_title("Estimation of EMC.")
    axs.plot(epochs, emc, "b-" )
    axs.set_xlabel("Epoch")
    axs.set_ylabel("EMC (max n)")
    plt.show()


def plot_moments():

    fig, ax = plt.subplots(1, 2, figsize=(15,7))
    fig.suptitle("1st and 2nd moments for lr= 0.1.")
    files = [
        "emc_data_09042024_114827.txt",
        "emc_data_09042024_114852.txt",
        "emc_data_09042024_115051.txt"
    ]

    sample_size = [10, 100, 500]

    for file, ss in zip(files, sample_size):
        history = get_data_from_file("data/lr_testing/3/" + file)
        ax[0].plot(history["first_mom"], label="size= " + str(ss))
        ax[1].plot(history["second_mom"], label="size= " + str(ss))
        print(history["setup"])

    ax[0].legend()
    ax[0].set_title("First moment.")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Norm of moment.")

    ax[1].legend()
    ax[1].set_title("Second moment.")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Norm of moment.")

    plt.show()


def plot_learning_rates():

    files = [

    ]

    learning_rates = [0.001, 0.01, 0.1]

    for file, lr in zip(files, learning_rates):
        history = get_data_from_file("data/lr_testing/sample_500/" + file)
        plt.plot(history["mean_loss"], label="lr= " + str(lr))

    plt.ylabel("MSE")
    plt.xlabel("Epoch")
    plt.title(f'Sample size {history["sample_size"]}')
    plt.grid()
    plt.yscale("log")
    plt.legend()
    plt.show()


def plot_mean_ci(history):
    print("SETUP:\n", history["setup"], end="")

    mean_training_loss = history["mean_loss"]
    y_ci = history["y_ci"]
    sample_size = history["sample_size"]
    epochs = range(1, len(mean_training_loss) + 1)

    plt.plot([history["std_noise"] for i in epochs], linestyle="--", color="green", label="zero mark")

    plt.plot(mean_training_loss, color="blue", label="Mean training loss")

    plt.plot(y_ci, color="red", label="95% confidence interval (low)")

    plt.title(f"Data set size {sample_size}.")

    x_ticks = list(range(0, len(mean_training_loss) + 1, 20))
    x_ticks[0] = 1

    x_ticks_labels = [str(i) for i in x_ticks]

    plt.xticks(x_ticks, x_ticks_labels)
    plt.ylabel("Training loss (MSE)")
    plt.yscale("symlog")
    plt.xlabel("Epochs")
    plt.grid()
    plt.legend()
    plt.show()


def plot_training():
    files = [
        "/home/anza/kanidiatarbete/emc_estimation/test1/data/emc_data_16042024_162416.txt"
    ]

    print("Plotting...")
    for file in files:
        history = get_data_from_file(file)
        plot_mean_ci(history)


def main():
    plt.rcParams.update({'font.size': 15})
    print("Plotting...")

    # plot_moments()
    # plot_learning_rates()
    # plot_training()
    plot_emc_per_epoch()
    print("DONE!")


if __name__ == '__main__':
    main()
