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


class Net_test(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 5)
        self.lin2 = nn.Linear(5, 2)
        self.lin3 = nn.Linear(2, 1)

    def forward(self, x):
        z1 = torch.relu(self.lin1(x))
        z2 = torch.relu(self.lin2(z1))
        out = self.lin3(z2)
        return out

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

        batch_learnable_params = True
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


#                    Net2
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


class Net2(nn.Module):
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

        batch_learnable_params = True
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


#                    Net3
#
#                      x
#                      | ---------linear(5,20)---
#                 linear(5, 10)                 |
#                  batch norm                   |
#                     relu                      |
#                 linear(10, 10)                |
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
#                 linear(40,40)                 |
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
#                 linear(10, 10)                |
#                  batch norm                   |
#                     relu                      |
#                 linear(10,5)                  |
#                  batch norm                   |
#                      | + <---------------------
#                     relu
#                 linear(5,2)
#                  batch norm
#                     relu
#                 linear(2,2)
#                  batch norm
#                     relu
#                 linear(2,1)
#                      |
#                      y


class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 10)
        self.lin3 = nn.Linear(10, 20)

        self.lin4 = nn.Linear(20, 40)
        self.lin5 = nn.Linear(40, 40)
        self.lin6 = nn.Linear(40, 20)

        self.lin7 = nn.Linear(20, 10)
        self.lin8 = nn.Linear(10, 10)
        self.lin9 = nn.Linear(10, 5)

        self.lin10 = nn.Linear(5, 2)
        self.lin11 = nn.Linear(2, 2)
        self.lin12 = nn.Linear(2, 1)

        batch_learnable_params = True
        self.bn1 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn2 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn3 = nn.BatchNorm1d(20, affine=batch_learnable_params)

        self.bn4 = nn.BatchNorm1d(40, affine=batch_learnable_params)
        self.bn5 = nn.BatchNorm1d(40, affine=batch_learnable_params)
        self.bn6 = nn.BatchNorm1d(20, affine=batch_learnable_params)

        self.bn7 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn8 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn9 = nn.BatchNorm1d(5, affine=batch_learnable_params)
        self.bn10 = nn.BatchNorm1d(2, affine=batch_learnable_params)
        self.bn11 = nn.BatchNorm1d(2, affine=batch_learnable_params)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 20)
        self.skip3 = nn.Linear(20, 5)

    def forward(self, x):
        skip_1 = self.skip1(x)
        z1 = torch.relu(self.bn1(self.lin1(x)))
        z2 = torch.relu(self.bn2(self.lin2(z1)))
        z3 = torch.relu(self.bn3(self.lin3(z2)) + skip_1)

        skip_2 = self.skip2(z3)
        z4 = torch.relu(self.bn4(self.lin4(z3)))
        z5 = torch.relu(self.bn5(self.lin5(z4)))
        z6 = torch.relu(self.bn6(self.lin6(z5)) + skip_2)

        skip_3 = self.skip3(z6)
        z7 = torch.relu(self.bn7(self.lin7(z6)))
        z8 = torch.relu(self.bn8(self.lin8(z7)))
        z9 = torch.relu(self.bn9(self.lin9(z8)) + skip_3)

        z10 = torch.relu(self.bn10(self.lin10(z9)))
        z11 = torch.relu(self.bn11(self.lin11(z10)))
        out = self.lin12(z11)
        return out


#                    Net4
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
#                      | ---------linear(20,80)--
#                 linear(20,40)                 |
#                  batch norm                   |
#                     relu                      |
#                 linear(40,80)                 |
#                  batch norm                   |
#                      | + <---------------------
#                     relu
#                      | ---------linear(80,20)--
#                 linear(80,40)                 |
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


class Net4(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)

        self.lin3 = nn.Linear(20, 40)
        self.lin4 = nn.Linear(40, 80)

        self.lin5 = nn.Linear(80, 40)
        self.lin6 = nn.Linear(40, 20)

        self.lin7 = nn.Linear(20, 10)
        self.lin8 = nn.Linear(10, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        batch_learnable_params = True
        self.bn1 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn2 = nn.BatchNorm1d(20, affine=batch_learnable_params)
        self.bn3 = nn.BatchNorm1d(40, affine=batch_learnable_params)
        self.bn4 = nn.BatchNorm1d(80, affine=batch_learnable_params)
        self.bn5 = nn.BatchNorm1d(40, affine=batch_learnable_params)
        self.bn6 = nn.BatchNorm1d(20, affine=batch_learnable_params)
        self.bn7 = nn.BatchNorm1d(10, affine=batch_learnable_params)
        self.bn8 = nn.BatchNorm1d(5, affine=batch_learnable_params)
        self.bn9 = nn.BatchNorm1d(2, affine=batch_learnable_params)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 80)
        self.skip3 = nn.Linear(80, 20)
        self.skip4 = nn.Linear(20, 5)

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