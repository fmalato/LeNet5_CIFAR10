from torch import nn
from collections import OrderedDict


class LeNet5(nn.Module):

    def __init__(self, num_channels=3):
        super().__init__()
        self.__name__ = 'LeNet5'

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(num_channels, 6, kernel_size=(5, 5))),
            ('relu1', nn.Tanh()),
            ('s2', nn.MaxPool2d(kernel_size=(1, 1), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.Tanh()),
            ('s4', nn.MaxPool2d(kernel_size=(1, 1), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.Tanh())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.Tanh()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


class LeNet5DyadicConv(nn.Module):

    def __init__(self, num_channels=3):
        super().__init__()
        self.__name__ = 'LeNet5DyadicConv'

        self.convnet = nn.Sequential(OrderedDict([
            ('16x16', ConvBlock(num_channels, 64, kernel_size=(3, 3), padding=1)),
            ('8x8', ConvBlock(64, 64, kernel_size=(3, 3), padding=1)),
            ('4x4', ConvBlock(64, 64, kernel_size=(3, 3), padding=1)),
            ('2x2', ConvBlock(64, 64, kernel_size=(1, 1))),
            ('1x1', ConvBlock(64, 64, kernel_size=(1, 1))),
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(64, 64)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(64, 10)),
            ('sig8', nn.LogSoftmax())
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=0):
        super().__init__()
        self.convBlock = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            ('relu', nn.ReLU()),
            ('pool', nn.MaxPool2d(kernel_size=(1, 1), stride=2, padding=0))
        ]))

    def forward(self, x):
        return self.convBlock(x)
