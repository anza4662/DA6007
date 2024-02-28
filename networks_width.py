import torch
import torch.nn as nn


class NetW1(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 5)
        self.lin2 = nn.Linear(5, 5)
        self.lin3 = nn.Linear(5, 5)
        self.lin4 = nn.Linear(5, 5)
        self.lin5 = nn.Linear(5, 5)
        self.lin6 = nn.Linear(5, 5)
        self.lin7 = nn.Linear(5, 5)
        self.lin8 = nn.Linear(5, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(5)
        self.bn2 = nn.BatchNorm1d(5)
        self.bn3 = nn.BatchNorm1d(5)
        self.bn4 = nn.BatchNorm1d(5)
        self.bn5 = nn.BatchNorm1d(5)
        self.bn6 = nn.BatchNorm1d(5)
        self.bn7 = nn.BatchNorm1d(5)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, 5)
        self.skip2 = nn.Linear(5, 5)
        self.skip3 = nn.Linear(5, 5)
        self.skip4 = nn.Linear(5, 5)

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


class NetW2(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 10)
        self.lin3 = nn.Linear(10, 10)
        self.lin4 = nn.Linear(10, 10)
        self.lin5 = nn.Linear(10, 10)
        self.lin6 = nn.Linear(10, 10)
        self.lin7 = nn.Linear(10, 10)
        self.lin8 = nn.Linear(10, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(10)
        self.bn3 = nn.BatchNorm1d(10)
        self.bn4 = nn.BatchNorm1d(10)
        self.bn5 = nn.BatchNorm1d(10)
        self.bn6 = nn.BatchNorm1d(10)
        self.bn7 = nn.BatchNorm1d(10)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, 10)
        self.skip2 = nn.Linear(10, 10)
        self.skip3 = nn.Linear(10, 10)
        self.skip4 = nn.Linear(10, 5)

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


class NetW3(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 15)
        self.lin3 = nn.Linear(15, 15)
        self.lin4 = nn.Linear(15, 15)
        self.lin5 = nn.Linear(15, 15)
        self.lin6 = nn.Linear(15, 15)
        self.lin7 = nn.Linear(15, 10)
        self.lin8 = nn.Linear(10, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(15)
        self.bn3 = nn.BatchNorm1d(15)
        self.bn4 = nn.BatchNorm1d(15)
        self.bn5 = nn.BatchNorm1d(15)
        self.bn6 = nn.BatchNorm1d(15)
        self.bn7 = nn.BatchNorm1d(10)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, 15)
        self.skip2 = nn.Linear(15, 15)
        self.skip3 = nn.Linear(15, 15)
        self.skip4 = nn.Linear(15, 5)

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


class NetW4(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)
        self.lin3 = nn.Linear(20, 20)
        self.lin4 = nn.Linear(20, 20)
        self.lin5 = nn.Linear(20, 20)
        self.lin6 = nn.Linear(20, 20)
        self.lin7 = nn.Linear(20, 10)
        self.lin8 = nn.Linear(10, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(20)
        self.bn4 = nn.BatchNorm1d(20)
        self.bn5 = nn.BatchNorm1d(20)
        self.bn6 = nn.BatchNorm1d(20)
        self.bn7 = nn.BatchNorm1d(10)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 20)
        self.skip3 = nn.Linear(20, 20)
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


class NetW5(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)
        self.lin3 = nn.Linear(20, 25)
        self.lin4 = nn.Linear(25, 25)
        self.lin5 = nn.Linear(25, 25)
        self.lin6 = nn.Linear(25, 20)
        self.lin7 = nn.Linear(20, 10)
        self.lin8 = nn.Linear(10, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(25)
        self.bn4 = nn.BatchNorm1d(25)
        self.bn5 = nn.BatchNorm1d(25)
        self.bn6 = nn.BatchNorm1d(20)
        self.bn7 = nn.BatchNorm1d(10)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 25)
        self.skip3 = nn.Linear(25, 20)
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


class NetW6(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)
        self.lin3 = nn.Linear(20, 30)
        self.lin4 = nn.Linear(30, 30)
        self.lin5 = nn.Linear(30, 30)
        self.lin6 = nn.Linear(30, 20)
        self.lin7 = nn.Linear(20, 10)
        self.lin8 = nn.Linear(10, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(30)
        self.bn4 = nn.BatchNorm1d(30)
        self.bn5 = nn.BatchNorm1d(30)
        self.bn6 = nn.BatchNorm1d(20)
        self.bn7 = nn.BatchNorm1d(10)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 30)
        self.skip3 = nn.Linear(30, 20)
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


class NetW7(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)
        self.lin3 = nn.Linear(20, 35)
        self.lin4 = nn.Linear(35, 35)
        self.lin5 = nn.Linear(35, 35)
        self.lin6 = nn.Linear(35, 20)
        self.lin7 = nn.Linear(20, 10)
        self.lin8 = nn.Linear(10, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(35)
        self.bn4 = nn.BatchNorm1d(35)
        self.bn5 = nn.BatchNorm1d(35)
        self.bn6 = nn.BatchNorm1d(20)
        self.bn7 = nn.BatchNorm1d(10)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 35)
        self.skip3 = nn.Linear(35, 20)
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


class NetW8(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)
        self.lin3 = nn.Linear(20, 40)
        self.lin4 = nn.Linear(40, 40)
        self.lin5 = nn.Linear(40, 40)
        self.lin6 = nn.Linear(40, 20)
        self.lin7 = nn.Linear(20, 10)
        self.lin8 = nn.Linear(10, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(40)
        self.bn4 = nn.BatchNorm1d(40)
        self.bn5 = nn.BatchNorm1d(40)
        self.bn6 = nn.BatchNorm1d(20)
        self.bn7 = nn.BatchNorm1d(10)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 40)
        self.skip3 = nn.Linear(40, 20)
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


class NetW9(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)
        self.lin3 = nn.Linear(20, 40)
        self.lin4 = nn.Linear(40, 45)
        self.lin5 = nn.Linear(45, 40)
        self.lin6 = nn.Linear(40, 20)
        self.lin7 = nn.Linear(20, 10)
        self.lin8 = nn.Linear(10, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(40)
        self.bn4 = nn.BatchNorm1d(45)
        self.bn5 = nn.BatchNorm1d(40)
        self.bn6 = nn.BatchNorm1d(20)
        self.bn7 = nn.BatchNorm1d(10)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 45)
        self.skip3 = nn.Linear(45, 20)
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


class NetW10(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)
        self.lin3 = nn.Linear(20, 40)
        self.lin4 = nn.Linear(40, 50)
        self.lin5 = nn.Linear(50, 40)
        self.lin6 = nn.Linear(40, 20)
        self.lin7 = nn.Linear(20, 10)
        self.lin8 = nn.Linear(10, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(40)
        self.bn4 = nn.BatchNorm1d(50)
        self.bn5 = nn.BatchNorm1d(40)
        self.bn6 = nn.BatchNorm1d(20)
        self.bn7 = nn.BatchNorm1d(10)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 50)
        self.skip3 = nn.Linear(50, 20)
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


class NetW11(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)
        self.lin3 = nn.Linear(20, 40)
        self.lin4 = nn.Linear(40, 55)
        self.lin5 = nn.Linear(55, 40)
        self.lin6 = nn.Linear(40, 20)
        self.lin7 = nn.Linear(20, 10)
        self.lin8 = nn.Linear(10, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(40)
        self.bn4 = nn.BatchNorm1d(55)
        self.bn5 = nn.BatchNorm1d(40)
        self.bn6 = nn.BatchNorm1d(20)
        self.bn7 = nn.BatchNorm1d(10)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 55)
        self.skip3 = nn.Linear(55, 20)
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


class NetW12(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)
        self.lin3 = nn.Linear(20, 40)
        self.lin4 = nn.Linear(40, 60)
        self.lin5 = nn.Linear(60, 40)
        self.lin6 = nn.Linear(40, 20)
        self.lin7 = nn.Linear(20, 10)
        self.lin8 = nn.Linear(10, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(40)
        self.bn4 = nn.BatchNorm1d(60)
        self.bn5 = nn.BatchNorm1d(40)
        self.bn6 = nn.BatchNorm1d(20)
        self.bn7 = nn.BatchNorm1d(10)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 60)
        self.skip3 = nn.Linear(60, 20)
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
    
    
class NetW13(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)
        self.lin3 = nn.Linear(20, 40)
        self.lin4 = nn.Linear(40, 65)
        self.lin5 = nn.Linear(65, 40)
        self.lin6 = nn.Linear(40, 20)
        self.lin7 = nn.Linear(20, 10)
        self.lin8 = nn.Linear(10, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(40)
        self.bn4 = nn.BatchNorm1d(65)
        self.bn5 = nn.BatchNorm1d(40)
        self.bn6 = nn.BatchNorm1d(20)
        self.bn7 = nn.BatchNorm1d(10)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 65)
        self.skip3 = nn.Linear(65, 20)
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
    

class NetW14(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)
        self.lin3 = nn.Linear(20, 40)
        self.lin4 = nn.Linear(40, 70)
        self.lin5 = nn.Linear(70, 40)
        self.lin6 = nn.Linear(40, 20)
        self.lin7 = nn.Linear(20, 10)
        self.lin8 = nn.Linear(10, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(40)
        self.bn4 = nn.BatchNorm1d(70)
        self.bn5 = nn.BatchNorm1d(40)
        self.bn6 = nn.BatchNorm1d(20)
        self.bn7 = nn.BatchNorm1d(10)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 70)
        self.skip3 = nn.Linear(70, 20)
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
    

class NetW15(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(5, 10)
        self.lin2 = nn.Linear(10, 20)
        self.lin3 = nn.Linear(20, 40)
        self.lin4 = nn.Linear(40, 75)
        self.lin5 = nn.Linear(75, 40)
        self.lin6 = nn.Linear(40, 20)
        self.lin7 = nn.Linear(20, 10)
        self.lin8 = nn.Linear(10, 5)

        self.lin9 = nn.Linear(5, 2)
        self.lin10 = nn.Linear(2, 1)

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(40)
        self.bn4 = nn.BatchNorm1d(75)
        self.bn5 = nn.BatchNorm1d(40)
        self.bn6 = nn.BatchNorm1d(20)
        self.bn7 = nn.BatchNorm1d(10)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

        self.skip1 = nn.Linear(5, 20)
        self.skip2 = nn.Linear(20, 75)
        self.skip3 = nn.Linear(75, 20)
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
    

class NetW16(nn.Module):
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

        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(20)
        self.bn3 = nn.BatchNorm1d(40)
        self.bn4 = nn.BatchNorm1d(80)
        self.bn5 = nn.BatchNorm1d(40)
        self.bn6 = nn.BatchNorm1d(20)
        self.bn7 = nn.BatchNorm1d(10)
        self.bn8 = nn.BatchNorm1d(5)
        self.bn9 = nn.BatchNorm1d(2)

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
