import torch
import torch.nn as nn

#TODO:
# Data (Parameter for non-linearity)                    DONE
# Noise variance                                        DONE
# Architecture                                          DONE
# Batch normalization                                   DONE
# Skip connections                                      SORT OF
# Initialization (How is the network initialized)       DONE
# Weight distribution                                   DONE
# Adam parameters                                       DONE

#                     Net1
#
#                      x
#                      | -------------linear(5,20)---
#                  linear(5, 10)                    |
#                   batch norm                      |
#                     relu                          |
#                  linear(10,20)                    |
#                   batch norm                      |
#                      | + <-------------------------
#                     relu
#                      |  ------------linear(20,5)---
#                  linear(20,10)                    |
#                   batch norm                      |
#                     relu                          |
#                  linear(10,5)                     |
#                   batch norm                      |
#                      | + <-------------------------
#                     relu
#                  linear(5,2)
#                   batch norm
#                     relu
#                  linear(2,1)
#                      |
#                      y


class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)

        self.lin3 = nn.Linear(20, 10)
        self.lin4 = nn.Linear(10, 5)

        self.lin5 = nn.Linear(5, 2)
        self.lin6 = nn.Linear(2, 1)

        batch_learnable_params = False
        self.bn1 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn2 = nn.BatchNorm1d(20, affine=batch_learnable_params)
        self.bn3 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn4 = nn.BatchNorm1d(5, affine=batch_learnable_params)
        self.bn5 = nn.BatchNorm1d(2, affine=batch_learnable_params)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 5)

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


#                     Net2
#
#                      x
#                      |
#                 linear(5, 10)
#                  batch norm
#                     relu
#                      | ----------------
#                 linear(10,20)         |
#                   batch norm          |
#                     relu              |
#                  linear(20,10)        |
#                  batch norm           |
#                      | + <-------------
#                     relu
#                      | ------linear(10,2)--
#                 linear(10,5)              |
#                  batch norm               |
#                      relu                 |
#                 linear(5,2)               |
#                  batch norm               |
#                      | + <-----------------
#                     relu
#                 linear(2,1)
#                      |
#                      y


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)

        self.lin3 = nn.Linear(20, 10)
        self.lin4 = nn.Linear(10, 5)

        self.lin5 = nn.Linear(5, 2)
        self.lin6 = nn.Linear(2, 1)

        batch_learnable_params = False
        self.bn1 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn2 = nn.BatchNorm1d(20, affine=batch_learnable_params)
        self.bn3 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn4 = nn.BatchNorm1d(5, affine=batch_learnable_params)
        self.bn5 = nn.BatchNorm1d(2, affine=batch_learnable_params)

        self.skip = nn.Linear(10, 2)

    def forward(self, x):
        z1 = self.lin1(x)
        z2 = torch.relu(self.bn1(z1))

        skip_no_lin = z2

        z3 = self.lin2(z2)
        z4 = torch.relu(self.bn2(z3))
        z5 = self.lin3(z4)
        z6 = torch.relu(self.bn3(z5) + skip_no_lin)

        skip_with_lin = self.skip(z6)

        z7 = self.lin4(z6)
        z8 = torch.relu(self.bn4(z7))
        z9 = self.lin5(z8)
        z10 = torch.relu(self.bn5(z9) + skip_with_lin)

        out = self.lin6(z10)
        return out


#                    Net3
#
#                      x
#                      | ---------linear(5,20)---
#                 linear(5, 10)                 |
#                  batch norm                   |
#                     relu                      |
#                 linear(10, 20)                |
#                  batch norm                   |
#                      | + <--------------------
#                     relu
#                      | ---------linear(20,20)--
#                 linear(20,40)                 |
#                  batch norm                   |
#                     relu                      |
#                 linear(40,20)                 |
#                  batch norm                   |
#                      | + <---------------------
#                     relu
#                      | ---------linear(20,5)---
#                 linear(20, 10)                |
#                  batch norm                   |
#                     relu                      |
#                 linear(10,5)                  |
#                  batch norm                   |
#                      | + <---------------------
#                     relu
#                 linear(5,2)
#                  batch norm
#                     relu
#                 linear(2,1)
#                      |
#                      y


class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)

        self.lin3 = nn.Linear(20, 40)
        self.lin4 = nn.Linear(40, 20)

        self.lin5 = nn.Linear(20, 10)
        self.lin6 = nn.Linear(10, 5)

        self.lin7 = nn.Linear(5, 2)
        self.lin8 = nn.Linear(2, 1)

        batch_learnable_params = False
        self.bn1 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn2 = nn.BatchNorm1d(20, affine=batch_learnable_params)
        self.bn3 = nn.BatchNorm1d(40, affine=batch_learnable_params)
        self.bn4 = nn.BatchNorm1d(20, affine=batch_learnable_params)
        self.bn5 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn6 = nn.BatchNorm1d(5, affine=batch_learnable_params)
        self.bn7 = nn.BatchNorm1d(2, affine=batch_learnable_params)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 20)
        self.skip3 = nn.Linear(20, 5)

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
        z7 = torch.relu(self.bn7(self.lin7(z6)))
        z8 = self.lin8(z7)
        return z8


#                     Net4
#
#                      x
#                      | ----------------
#                 linear(5, 10)         |
#                  batch norm           |
#                     relu              |
#           --------   |                |
#           |      linear(10,20)        |
#           |      batch norm           |
#           |         relu              |
#           |      linear(20,10)        |
#           |      batch norm           |
#           -------> + |                |
#                     relu              |
#                  linear(10,5)         |
#                  batch norm           |
#                      | + <-------------
#                     relu
#                  linear(5,2)
#                  batch norm
#                     relu
#                  linear(2,1)
#                      |
#                      y


class Net4(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)

        self.lin3 = nn.Linear(20, 10)
        self.lin4 = nn.Linear(10, 5)

        self.lin5 = nn.Linear(5, 2)
        self.lin6 = nn.Linear(2, 1)

        batch_learnable_params = False
        self.bn1 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn2 = nn.BatchNorm1d(20, affine=batch_learnable_params)
        self.bn3 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn4 = nn.BatchNorm1d(5, affine=batch_learnable_params)
        self.bn5 = nn.BatchNorm1d(2, affine=batch_learnable_params)

    def forward(self, x):
        skip1 = x

        z1 = self.lin1(x)
        z2 = torch.relu(self.bn1(z1))

        skip2 = z2

        z3 = self.lin2(z2)
        z4 = torch.relu(self.bn2(z3))
        z5 = self.lin3(z4)
        z6 = torch.relu(self.bn3(z5) + skip2)

        z7 = self.lin4(z6)
        z8 = torch.relu(self.bn4(z7) + skip1)
        z9 = self.lin5(z8)
        z10 = torch.relu(self.bn5(z9))

        out = self.lin6(z10)
        return out


#               Net5
#
#       Regular feedforward network
#          with architecture:
#         [5,10,20,10,5,2,1]
#


class Net5(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)
        self.lin3 = nn.Linear(20, 10)
        self.lin4 = nn.Linear(10, 5)
        self.lin5 = nn.Linear(5, 2)
        self.lin6 = nn.Linear(2, 1)

        batch_learnable_params = False
        self.bn1 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn2 = nn.BatchNorm1d(20, affine=batch_learnable_params)
        self.bn3 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn4 = nn.BatchNorm1d(5, affine=batch_learnable_params)
        self.bn5 = nn.BatchNorm1d(2, affine=batch_learnable_params)

    def forward(self, x):
        z1 = self.lin1(x)
        z2 = torch.relu(self.bn1(z1))
        z3 = self.lin2(z2)
        z4 = torch.relu(self.bn2(z3))
        z5 = self.lin3(z4)
        z6 = torch.relu(self.bn3(z5))
        z7 = self.lin4(z6)
        z8 = torch.relu(self.bn4(z7))
        z9 = self.lin5(z8)
        z10 = torch.relu(self.bn5(z9))
        out = self.lin6(z10)
        return out


#               Net6
#
#       Regular feedforward network
#          with architecture:
#           [5,10,5,2,1]
#


class Net6(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 5)

        self.lin3 = nn.Linear(5, 2)
        self.lin4 = nn.Linear(2, 1)

        batch_learnable_params = False
        self.bn1 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn2 = nn.BatchNorm1d(5, affine=batch_learnable_params)
        self.bn3 = nn.BatchNorm1d(2, affine=batch_learnable_params)

    def forward(self, x):
        z1 = self.lin1(x)
        z2 = torch.relu(self.bn1(z1))
        z3 = self.lin2(z2)
        z4 = torch.relu(self.bn2(z3))
        z5 = self.lin3(z4)
        z6 = torch.relu(self.bn3(z5))
        out = self.lin4(z6)
        return out


#                     Net7
#
#                      x
#                      |  ---------------
#                  linear(5,10)         |
#                   batch norm          |
#                     relu              |
#                  linear(10,5)         |
#                   batch norm          |
#                      | + <-------------
#                     relu
#                  linear(5,2)
#                   batch norm
#                     relu
#                  linear(2,1)
#                      |
#                      y


class Net7(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 5)

        self.lin3 = nn.Linear(5, 2)
        self.lin4 = nn.Linear(2, 1)

        batch_learnable_params = False
        self.bn1 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn2 = nn.BatchNorm1d(5, affine=batch_learnable_params)
        self.bn3 = nn.BatchNorm1d(2, affine=batch_learnable_params)

    def forward(self, x):
        z1 = self.lin1(x)
        z2 = torch.relu(self.bn1(z1))
        z3 = self.lin2(z2)
        z4 = torch.relu(self.bn2(z3) + x)
        z5 = self.lin3(z4)
        z6 = torch.relu(self.bn3(z5))
        out = self.lin4(z6)
        return out
