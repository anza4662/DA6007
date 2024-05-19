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
        if len(hidden_layers) != 3:
            raise Exception("Invalid number of hidden layers. Must be 3.")

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
#                      | ---------linear(l3,l5)--
#                 linear(l3,l4)                 |
#                  batch norm                   |
#                     relu                      |
#                 linear(l4,l5)                 |
#                  batch norm                   |
#                      | + <---------------------
#                     relu
#                      | ---------linear(l5,5)---
#                 linear(l5, l6)                |
#                  batch norm                   |
#                     relu                      |
#                 linear(l6,5)                  |
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
    """ The medium class of networks. Takes a list of 7 hidden layers. """

    def __init__(self, hidden_layers):
        super().__init__()
        if len(hidden_layers) != 7:
            raise Exception("Invalid number of hidden layers. Must be 7.")

        self.lin1 = nn.Linear(5, hidden_layers[0])
        self.lin2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.lin3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.lin4 = nn.Linear(hidden_layers[2], hidden_layers[3])
        self.lin5 = nn.Linear(hidden_layers[3], hidden_layers[4])
        self.lin6 = nn.Linear(hidden_layers[4], hidden_layers[5])
        self.lin7 = nn.Linear(hidden_layers[5], hidden_layers[6])
        self.lin8 = nn.Linear(hidden_layers[6], 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.bn2 = nn.BatchNorm1d(hidden_layers[1])
        self.bn3 = nn.BatchNorm1d(hidden_layers[2])
        self.bn4 = nn.BatchNorm1d(hidden_layers[3])
        self.bn5 = nn.BatchNorm1d(hidden_layers[4])
        self.bn6 = nn.BatchNorm1d(hidden_layers[5])
        self.bn7 = nn.BatchNorm1d(hidden_layers[6])
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, hidden_layers[1])
        self.skip2 = nn.Linear(hidden_layers[1], hidden_layers[3])
        self.skip3 = nn.Linear(hidden_layers[3], hidden_layers[5])
        self.skip4 = nn.Linear(hidden_layers[5], 5)

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
#                      | --------linear(l5,l7)--
#                 linear(l5,l6)                |
#                  batch norm                  |
#                     relu                     |
#                 linear(l6,l7)                |
#                  batch norm                  |
#                      | + <--------------------
#                     relu
#                      | --------linear(l7,l9)--
#                 linear(l7,l8)                |
#                  batch norm                  |
#                     relu                     |
#                 linear(l8,l9)                |
#                  batch norm                  |
#                      | + <--------------------
#                     relu
#                      | ---------linear(l9,5)--
#                 linear(l9, l10)              |
#                  batch norm                  |
#                     relu                     |
#                 linear(l10,5)                |
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
    """ The Large class of networks. Takes a list of 11 hidden layers. """

    def __init__(self, hidden_layers):
        super().__init__()
        if len(hidden_layers) != 11:
            raise Exception("Invalid number of hidden layers. Must be 11.")

        self.lin1 = nn.Linear(5, hidden_layers[0])
        self.lin2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.lin3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.lin4 = nn.Linear(hidden_layers[2], hidden_layers[3])
        self.lin5 = nn.Linear(hidden_layers[3], hidden_layers[4])
        self.lin6 = nn.Linear(hidden_layers[4], hidden_layers[5])
        self.lin7 = nn.Linear(hidden_layers[5], hidden_layers[6])
        self.lin8 = nn.Linear(hidden_layers[6], hidden_layers[7])
        self.lin9 = nn.Linear(hidden_layers[7], hidden_layers[8])
        self.lin10 = nn.Linear(hidden_layers[8], hidden_layers[9])
        self.lin11 = nn.Linear(hidden_layers[9], hidden_layers[10])
        self.lin12 = nn.Linear(hidden_layers[10], 5)

        self.lin13 = nn.Linear(5, 2)
        self.lin14 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.bn2 = nn.BatchNorm1d(hidden_layers[1])
        self.bn3 = nn.BatchNorm1d(hidden_layers[2])
        self.bn4 = nn.BatchNorm1d(hidden_layers[3])
        self.bn5 = nn.BatchNorm1d(hidden_layers[4])
        self.bn6 = nn.BatchNorm1d(hidden_layers[5])
        self.bn7 = nn.BatchNorm1d(hidden_layers[6])
        self.bn8 = nn.BatchNorm1d(hidden_layers[7])
        self.bn9 = nn.BatchNorm1d(hidden_layers[8])
        self.bn10 = nn.BatchNorm1d(hidden_layers[9])
        self.bn11 = nn.BatchNorm1d(hidden_layers[10])
        self.bn12 = nn.BatchNorm1d(5)
        self.bn13 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, hidden_layers[1])
        self.skip2 = nn.Linear(hidden_layers[1], hidden_layers[3])
        self.skip3 = nn.Linear(hidden_layers[3], hidden_layers[5])
        self.skip4 = nn.Linear(hidden_layers[5], hidden_layers[7])
        self.skip5 = nn.Linear(hidden_layers[7], hidden_layers[9])
        self.skip6 = nn.Linear(hidden_layers[9], 5)

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


class NetHuge(nn.Module):
    """ The huuuuuge class of networks. Takes a list of 15 hidden layers. """

    def __init__(self, hidden_layers):
        super().__init__()
        if len(hidden_layers) != 15:
            raise Exception("Invalid number of hidden layers. Must be 15.")

        self.lin1 = nn.Linear(5, hidden_layers[0])
        self.lin2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.lin3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.lin4 = nn.Linear(hidden_layers[2], hidden_layers[3])
        self.lin5 = nn.Linear(hidden_layers[3], hidden_layers[4])
        self.lin6 = nn.Linear(hidden_layers[4], hidden_layers[5])
        self.lin7 = nn.Linear(hidden_layers[5], hidden_layers[6])
        self.lin8 = nn.Linear(hidden_layers[6], hidden_layers[7])
        self.lin9 = nn.Linear(hidden_layers[7], hidden_layers[8])
        self.lin10 = nn.Linear(hidden_layers[8], hidden_layers[9])
        self.lin11 = nn.Linear(hidden_layers[9], hidden_layers[10])
        self.lin12 = nn.Linear(hidden_layers[10], hidden_layers[11])
        self.lin13 = nn.Linear(hidden_layers[11], hidden_layers[12])
        self.lin14 = nn.Linear(hidden_layers[12], hidden_layers[13])
        self.lin15 = nn.Linear(hidden_layers[13], hidden_layers[14])
        self.lin16 = nn.Linear(hidden_layers[14], 5)

        self.lin17 = nn.Linear(5, 2)
        self.lin18 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.bn2 = nn.BatchNorm1d(hidden_layers[1])
        self.bn3 = nn.BatchNorm1d(hidden_layers[2])
        self.bn4 = nn.BatchNorm1d(hidden_layers[3])
        self.bn5 = nn.BatchNorm1d(hidden_layers[4])
        self.bn6 = nn.BatchNorm1d(hidden_layers[5])
        self.bn7 = nn.BatchNorm1d(hidden_layers[6])
        self.bn8 = nn.BatchNorm1d(hidden_layers[7])
        self.bn9 = nn.BatchNorm1d(hidden_layers[8])
        self.bn10 = nn.BatchNorm1d(hidden_layers[9])
        self.bn11 = nn.BatchNorm1d(hidden_layers[10])
        self.bn12 = nn.BatchNorm1d(hidden_layers[11])
        self.bn13 = nn.BatchNorm1d(hidden_layers[12])
        self.bn14 = nn.BatchNorm1d(hidden_layers[13])
        self.bn15 = nn.BatchNorm1d(hidden_layers[14])
        self.bn16 = nn.BatchNorm1d(5)
        self.bn17 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, hidden_layers[1])
        self.skip2 = nn.Linear(hidden_layers[1], hidden_layers[3])
        self.skip3 = nn.Linear(hidden_layers[3], hidden_layers[5])
        self.skip4 = nn.Linear(hidden_layers[5], hidden_layers[7])
        self.skip5 = nn.Linear(hidden_layers[7], hidden_layers[9])
        self.skip6 = nn.Linear(hidden_layers[9], hidden_layers[11])
        self.skip7 = nn.Linear(hidden_layers[11], hidden_layers[13])
        self.skip8 = nn.Linear(hidden_layers[13], 5)

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

        skip_7 = self.skip7(z12)
        z13 = torch.relu(self.bn13(self.lin13(z12)))
        z14 = torch.relu(self.bn14(self.lin14(z13)) + skip_7)

        skip_8 = self.skip8(z14)
        z15 = torch.relu(self.bn15(self.lin15(z14)))
        z16 = torch.relu(self.bn16(self.lin16(z15)) + skip_8)

        z17 = torch.relu(self.bn17(self.lin17(z16)))
        out = self.lin18(z17)
        return out


class NetExtreme(nn.Module):
    """ The dumb class of networks. Takes a list of 27 hidden layers. """

    def __init__(self, hidden_layers):
        super().__init__()
        if len(hidden_layers) != 27:
            raise Exception("Invalid number of hidden layers. Must be 27.")

        self.lin1 = nn.Linear(5, hidden_layers[0])
        self.lin2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.lin3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        self.lin4 = nn.Linear(hidden_layers[2], hidden_layers[3])
        self.lin5 = nn.Linear(hidden_layers[3], hidden_layers[4])
        self.lin6 = nn.Linear(hidden_layers[4], hidden_layers[5])
        self.lin7 = nn.Linear(hidden_layers[5], hidden_layers[6])
        self.lin8 = nn.Linear(hidden_layers[6], hidden_layers[7])
        self.lin9 = nn.Linear(hidden_layers[7], hidden_layers[8])
        self.lin10 = nn.Linear(hidden_layers[8], hidden_layers[9])
        self.lin11 = nn.Linear(hidden_layers[9], hidden_layers[10])
        self.lin12 = nn.Linear(hidden_layers[10], hidden_layers[11])
        self.lin13 = nn.Linear(hidden_layers[11], hidden_layers[12])
        self.lin14 = nn.Linear(hidden_layers[12], hidden_layers[13])
        self.lin15 = nn.Linear(hidden_layers[13], hidden_layers[14])
        self.lin16 = nn.Linear(hidden_layers[14], hidden_layers[15])
        self.lin17 = nn.Linear(hidden_layers[15], hidden_layers[16])
        self.lin18 = nn.Linear(hidden_layers[16], hidden_layers[17])
        self.lin19 = nn.Linear(hidden_layers[17], hidden_layers[18])
        self.lin20 = nn.Linear(hidden_layers[18], hidden_layers[19])
        self.lin21 = nn.Linear(hidden_layers[19], hidden_layers[20])
        self.lin22 = nn.Linear(hidden_layers[20], hidden_layers[21])
        self.lin23 = nn.Linear(hidden_layers[21], hidden_layers[22])
        self.lin24 = nn.Linear(hidden_layers[22], hidden_layers[23])
        self.lin25 = nn.Linear(hidden_layers[23], hidden_layers[24])
        self.lin26 = nn.Linear(hidden_layers[24], hidden_layers[25])
        self.lin27 = nn.Linear(hidden_layers[25], hidden_layers[26])
        self.lin28 = nn.Linear(hidden_layers[26], 5)

        self.lin29 = nn.Linear(5, 2)
        self.lin30 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(hidden_layers[0])
        self.bn2 = nn.BatchNorm1d(hidden_layers[1])
        self.bn3 = nn.BatchNorm1d(hidden_layers[2])
        self.bn4 = nn.BatchNorm1d(hidden_layers[3])
        self.bn5 = nn.BatchNorm1d(hidden_layers[4])
        self.bn6 = nn.BatchNorm1d(hidden_layers[5])
        self.bn7 = nn.BatchNorm1d(hidden_layers[6])
        self.bn8 = nn.BatchNorm1d(hidden_layers[7])
        self.bn9 = nn.BatchNorm1d(hidden_layers[8])
        self.bn10 = nn.BatchNorm1d(hidden_layers[9])
        self.bn11 = nn.BatchNorm1d(hidden_layers[10])
        self.bn12 = nn.BatchNorm1d(hidden_layers[11])
        self.bn13 = nn.BatchNorm1d(hidden_layers[12])
        self.bn14 = nn.BatchNorm1d(hidden_layers[13])
        self.bn15 = nn.BatchNorm1d(hidden_layers[14])
        self.bn16 = nn.BatchNorm1d(hidden_layers[15])
        self.bn17 = nn.BatchNorm1d(hidden_layers[16])
        self.bn18 = nn.BatchNorm1d(hidden_layers[17])
        self.bn19 = nn.BatchNorm1d(hidden_layers[18])
        self.bn20 = nn.BatchNorm1d(hidden_layers[19])
        self.bn21 = nn.BatchNorm1d(hidden_layers[20])
        self.bn22 = nn.BatchNorm1d(hidden_layers[21])
        self.bn23 = nn.BatchNorm1d(hidden_layers[22])
        self.bn24 = nn.BatchNorm1d(hidden_layers[23])
        self.bn25 = nn.BatchNorm1d(hidden_layers[24])
        self.bn26 = nn.BatchNorm1d(hidden_layers[25])
        self.bn27 = nn.BatchNorm1d(hidden_layers[26])
        self.bn28 = nn.BatchNorm1d(5)
        self.bn29 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, hidden_layers[1])
        self.skip2 = nn.Linear(hidden_layers[1], hidden_layers[3])
        self.skip3 = nn.Linear(hidden_layers[3], hidden_layers[5])
        self.skip4 = nn.Linear(hidden_layers[5], hidden_layers[7])
        self.skip5 = nn.Linear(hidden_layers[7], hidden_layers[9])
        self.skip6 = nn.Linear(hidden_layers[9], hidden_layers[11])
        self.skip7 = nn.Linear(hidden_layers[11], hidden_layers[13])
        self.skip8 = nn.Linear(hidden_layers[13], hidden_layers[15])
        self.skip9 = nn.Linear(hidden_layers[15], hidden_layers[17])
        self.skip10 = nn.Linear(hidden_layers[17], hidden_layers[19])
        self.skip11 = nn.Linear(hidden_layers[19], hidden_layers[21])
        self.skip12 = nn.Linear(hidden_layers[21], hidden_layers[23])
        self.skip13 = nn.Linear(hidden_layers[23], hidden_layers[25])
        self.skip14 = nn.Linear(hidden_layers[25], 5)

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

        skip_7 = self.skip7(z12)
        z13 = torch.relu(self.bn13(self.lin13(z12)))
        z14 = torch.relu(self.bn14(self.lin14(z13)) + skip_7)

        skip_8 = self.skip8(z14)
        z15 = torch.relu(self.bn15(self.lin15(z14)))
        z16 = torch.relu(self.bn16(self.lin16(z15)) + skip_8)

        skip_9 = self.skip9(z16)
        z17 = torch.relu(self.bn17(self.lin17(z16)))
        z18 = torch.relu(self.bn18(self.lin18(z17)) + skip_9)

        skip_10 = self.skip10(z18)
        z19 = torch.relu(self.bn19(self.lin19(z18)))
        z20 = torch.relu(self.bn20(self.lin20(z19)) + skip_10)

        skip_11 = self.skip11(z20)
        z21 = torch.relu(self.bn21(self.lin21(z20)))
        z22 = torch.relu(self.bn22(self.lin22(z21)) + skip_11)

        skip_12 = self.skip12(z22)
        z23 = torch.relu(self.bn23(self.lin23(z22)))
        z24 = torch.relu(self.bn24(self.lin24(z23)) + skip_12)

        skip_13 = self.skip13(z24)
        z25 = torch.relu(self.bn25(self.lin25(z24)))
        z26 = torch.relu(self.bn26(self.lin26(z25)) + skip_13)

        skip_14 = self.skip14(z26)
        z27 = torch.relu(self.bn27(self.lin27(z26)))
        z28 = torch.relu(self.bn28(self.lin28(z27)) + skip_14)

        z29 = torch.relu(self.bn29(self.lin29(z28)))
        out = self.lin30(z29)

        return out