import torch
import torch.nn as nn


#                     NetSmall
#
#                      x
#                      | -------------linear(5,l1)----
#                  linear(5, l0)                     |
#                   batch norm                       |
#                     relu                           |
#                  linear(l0,l1)                     |
#                   batch norm                       |
#                      | + <--------------------------
#                     relu
#                      |  ------------linear(l1,5)----
#                  linear(l1,l2)                     |
#                   batch norm                       |
#                     relu                           |
#                  linear(l2,5)                      |
#                   batch norm                       |
#                      | + <--------------------------
#                     relu
#                  linear(5,2)
#                   batch norm
#                     relu
#                  linear(2,1)
#                      |
#                      y


class NetSmall(nn.Module):
    """ The small class of networks. Takes a list of 3 hidden layers. """

    def __init__(self, hidden_layers):
        super().__init__()
        self.lin1 = nn.Linear(5, hidden_layers[0])
        self.lin2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.lin3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.lin4 = nn.Linear(hidden_layers[2], 5)

        self.lin5 = nn.Linear(5, 2)
        self.lin6 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.bn2 = nn.BatchNorm1d(hidden_layers[1])
        self.bn3 = nn.BatchNorm1d(hidden_layers[2])
        self.bn4 = nn.BatchNorm1d(5)
        self.bn5 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, hidden_layers[1])
        self.skip2 = nn.Linear(hidden_layers[1], 5)

    def forward(self, x):
        z1 = self.lin1(x)
        z_skip1 = self.skip1(x)
        z2 = torch.relu(self.bn1(z1))
        z3 = self.lin2(z2)
        z4 = torch.relu(self.bn2(z3) + z_skip1)

        z_skip2 = self.skip2(z4)
        z5 = self.lin3(z4)
        z6 = torch.relu(self.bn3(z5))
        z7 = self.lin4(z6)
        z8 = torch.relu(self.bn4(z7) + z_skip2)

        z9 = self.lin5(z8)
        z10 = torch.relu(self.bn5(z9))
        z11 = self.lin6(z10)
        return z11


#                    NetMedium
#
#                      x
#                      | ---------linear(5,l1)---
#                 linear(5, l0)                 |
#                  batch norm                   |
#                     relu                      |
#                 linear(l0, l1)                |
#                  batch norm                   |
#                      | + <--------------------
#                     relu
#                      | ---------linear(l1,l3)--
#                 linear(l1,l2)                 |
#                  batch norm                   |
#                     relu                      |
#                 linear(l2,l3)                 |
#                  batch norm                   |
#                      | + <---------------------
#                     relu
#                      | ---------linear(l3,l1)--
#                 linear(l3,l2)                 |
#                  batch norm                   |
#                     relu                      |
#                 linear(l2,l1)                 |
#                  batch norm                   |
#                      | + <---------------------
#                     relu
#                      | ---------linear(l1,5)---
#                 linear(l1, l0)                |
#                  batch norm                   |
#                     relu                      |
#                 linear(l0,5)                  |
#                  batch norm                   |
#                      | + <---------------------
#                     relu
#                 linear(5,2)
#                  batch norm
#                     relu
#                 linear(2,1)
#                      |
#                      y


class NetMedium(nn.Module):
    """ The medium class of networks. Takes a list of 4 hidden layers. """

    def __init__(self, hidden_layers):
        super().__init__()
        self.lin1 = nn.Linear(5, hidden_layers[0])
        self.lin2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.lin3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.lin4 = nn.Linear(hidden_layers[2], hidden_layers[3])
        self.lin5 = nn.Linear(hidden_layers[3], hidden_layers[2])
        self.lin6 = nn.Linear(hidden_layers[2], hidden_layers[1])
        self.lin7 = nn.Linear(hidden_layers[1], hidden_layers[0])
        self.lin8 = nn.Linear(hidden_layers[0], 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.bn2 = nn.BatchNorm1d(hidden_layers[1])
        self.bn3 = nn.BatchNorm1d(hidden_layers[2])
        self.bn4 = nn.BatchNorm1d(hidden_layers[3])
        self.bn5 = nn.BatchNorm1d(hidden_layers[2])
        self.bn6 = nn.BatchNorm1d(hidden_layers[1])
        self.bn7 = nn.BatchNorm1d(hidden_layers[0])
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, hidden_layers[1])
        self.skip2 = nn.Linear(hidden_layers[1], hidden_layers[3])
        self.skip3 = nn.Linear(hidden_layers[3], hidden_layers[1])
        self.skip4 = nn.Linear(hidden_layers[1], 5)

    def forward(self, x):
        skip_1 = self.skip1(x)
        z1 = torch.relu(self.bn1(self.lin1(x)))
        z2 = torch.relu(self.bn2(self.lin2(z1)) + skip_1)

        skip_2 = self.skip2(z2)
        z3 = torch.relu(self.bn3(self.lin3(z2)))
        z4 = torch.relu(self.bn4(self.lin4(z3)) + skip_2)

        skip_3 = self.skip3(z4)
        z5 = torch.relu(self.bn5(self.lin5(z4)))
        z6 = torch.relu(self.bn6(self.lin6(z5)) + skip_3)

        skip_4 = self.skip4(z6)
        z7 = torch.relu(self.bn7(self.lin7(z6)))
        z8 = torch.relu(self.bn8(self.lin8(z7)) + skip_4)

        z9 = torch.relu(self.bn9(self.lin9(z8)))
        out = self.lin10(z9)
        return out


#                    NetLarge
#
#                      x
#                      | --------linear(5,l1)---
#                 linear(5,l0)                 |
#                  batch norm                  |
#                     relu                     |
#                 linear(l0,l1)                |
#                  batch norm                  |
#                      | + <--------------------
#                     relu
#                      | --------linear(l1,l3)--
#                 linear(l1,l2)                |
#                  batch norm                  |
#                     relu                     |
#                 linear(l2,l3)                |
#                  batch norm                  |
#                      | + <--------------------
#                     relu
#                      | --------linear(l3,l5)--
#                 linear(l3,l4)                |
#                  batch norm                  |
#                     relu                     |
#                 linear(l4,l5)                |
#                  batch norm                  |
#                      | + <--------------------
#                     relu
#                      | --------linear(l5,l3)--
#                 linear(l5,l4)                |
#                  batch norm                  |
#                     relu                     |
#                 linear(l4,l3)                |
#                  batch norm                  |
#                      | + <--------------------
#                     relu
#                      | --------linear(l3,l1)--
#                 linear(l3,l2)                |
#                  batch norm                  |
#                     relu                     |
#                 linear(l2,l1)                |
#                  batch norm                  |
#                      | + <--------------------
#                     relu
#                      | ---------linear(l1,5)--
#                 linear(l1, l0)               |
#                  batch norm                  |
#                     relu                     |
#                 linear(l0,5)                 |
#                  batch norm                  |
#                      | + <--------------------
#                     relu
#                 linear(5,2)
#                  batch norm
#                     relu
#                 linear(2,1)
#                      |
#                      y


class NetLarge(nn.Module):
    """ The large class of networks. Takes a list of 6 hidden layers. """

    def __init__(self, hidden_layers):
        super().__init__()
        self.lin1 = nn.Linear(5, hidden_layers[0])
        self.lin2 = nn.Linear(hidden_layers[0], hidden_layers[1])

        self.lin3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.lin4 = nn.Linear(hidden_layers[2], hidden_layers[3])

        self.lin5 = nn.Linear(hidden_layers[3], hidden_layers[4])
        self.lin6 = nn.Linear(hidden_layers[4], hidden_layers[5])

        self.lin7 = nn.Linear(hidden_layers[5], hidden_layers[4])
        self.lin8 = nn.Linear(hidden_layers[4], hidden_layers[3])

        self.lin9 = nn.Linear(hidden_layers[3], hidden_layers[2])
        self.lin10 = nn.Linear(hidden_layers[2], hidden_layers[1])

        self.lin11 = nn.Linear(hidden_layers[1], hidden_layers[0])
        self.lin12 = nn.Linear(hidden_layers[0], 5)

        self.lin13 = nn.Linear(5, 2)
        self.lin14 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.bn2 = nn.BatchNorm1d(hidden_layers[1])
        self.bn3 = nn.BatchNorm1d(hidden_layers[2])
        self.bn4 = nn.BatchNorm1d(hidden_layers[3])
        self.bn5 = nn.BatchNorm1d(hidden_layers[4])
        self.bn6 = nn.BatchNorm1d(hidden_layers[5])
        self.bn7 = nn.BatchNorm1d(hidden_layers[4])
        self.bn8 = nn.BatchNorm1d(hidden_layers[3])
        self.bn9 = nn.BatchNorm1d(hidden_layers[2])
        self.bn10 = nn.BatchNorm1d(hidden_layers[1])
        self.bn11 = nn.BatchNorm1d(hidden_layers[0])
        self.bn12 = nn.BatchNorm1d(5)
        self.bn13 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, hidden_layers[1])
        self.skip2 = nn.Linear(hidden_layers[1], hidden_layers[3])
        self.skip3 = nn.Linear(hidden_layers[3], hidden_layers[5])
        self.skip4 = nn.Linear(hidden_layers[5], hidden_layers[3])
        self.skip5 = nn.Linear(hidden_layers[3], hidden_layers[1])
        self.skip6 = nn.Linear(hidden_layers[1], 5)

    def forward(self, x):
        skip_1 = self.skip1(x)
        z1 = torch.relu(self.bn1(self.lin1(x)))
        z2 = torch.relu(self.bn2(self.lin2(z1)) + skip_1)

        skip_2 = self.skip2(z2)
        z3 = torch.relu(self.bn3(self.lin3(z2)))
        z4 = torch.relu(self.bn4(self.lin4(z3)) + skip_2)

        skip_3 = self.skip3(z4)
        z5 = torch.relu(self.bn5(self.lin5(z4)))
        z6 = torch.relu(self.bn6(self.lin6(z5)) + skip_3)

        skip_4 = self.skip4(z6)
        z7 = torch.relu(self.bn7(self.lin7(z6)))
        z8 = torch.relu(self.bn8(self.lin8(z7)) + skip_4)

        skip_5 = self.skip5(z8)
        z9 = torch.relu(self.bn9(self.lin9(z8)))
        z10 = torch.relu(self.bn10(self.lin10(z9)) + skip_5)

        skip_6 = self.skip6(z10)
        z11 = torch.relu(self.bn11(self.lin11(z10)))
        z12 = torch.relu(self.bn12(self.lin12(z11)) + skip_6)

        z13 = torch.relu(self.bn13(self.lin13(z12)))
        out = self.lin14(z13)
        return out
