import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Some code taken from https://machinelearningmastery.com/building-a-regression-model-in-pytorch/

data = pd.read_csv("data_5_features_20k", index_col=0)
X = np.array(data.drop("val", axis=1))
y = np.array(data["val"])

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, shuffle=True)

scaler = StandardScaler()
scaler.fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)

train_X = torch.tensor(train_X, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32).reshape(-1, 1)

test_X = torch.tensor(test_X, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32).reshape(-1, 1)

size = 10
model = nn.Sequential(
    nn.Linear(5, size),
    nn.ReLU(),
    nn.Linear(size, size),
    nn.ReLU(),
    nn.Linear(size, size),
    nn.ReLU(),
    nn.Linear(size, size),
    nn.ReLU(),
    nn.Linear(size, size),
    nn.ReLU(),
    nn.Linear(size, 1),
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

n_epochs = 100
minibatch_size = 10
batch_start = torch.arange(0, len(train_X), minibatch_size)

best_mse = np.inf
best_weights = None
history = {
    "val_loss": [],
    "train_loss": []
}

for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
        bar.set_description(f"Epoch {epoch}")

        for start in bar:
            X_batch = train_X[start:start + minibatch_size]
            y_batch = train_y[start:start + minibatch_size]

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            bar.set_postfix(mse=float(loss))

    history["train_loss"].append(float(loss))

    # Validation
    model.eval()
    y_pred = model(test_X)
    val_mse = loss_fn(y_pred, test_y)
    val_mse = float(val_mse)
    history["val_loss"].append(val_mse)
    if val_mse < best_mse:
        best_mse = val_mse
        best_weights = copy.deepcopy(model.state_dict())

model.load_state_dict(best_weights)

print("Best MSE: %.2f" % best_mse)
plt.plot(history["val_loss"], label="val_loss")
plt.plot(history["train_loss"], label="train_loss")
plt.ylabel("MSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()
