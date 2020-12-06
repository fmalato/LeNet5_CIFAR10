import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchsummary as summary

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from net import LeNet5, LeNet5DyadicConv
from train import train, test


if __name__ == "__main__":
    # Network initialization
    lenet = LeNet5DyadicConv(num_channels=3)
    lenet.cuda()
    # Parameters initialization
    losses = []
    accuracies = []
    num_epochs = 500
    train_flag = False
    test_flag = True
    net_type = lenet.__name__
    lr = 1e-3
    train_split = 0.9
    test_split = 0.1
    batch_size = 128
    # Dataset & DataLoader initialization
    cifar = CIFAR10('/home/federico/PycharmProjects/LeNet_CIFAR10/datasets/CIFAR10/',
                    download=True,
                    transform=transforms.ToTensor()
                    )
    num_images = len(cifar)
    cifar = torch.utils.data.random_split(cifar, [int(num_images * train_split), int(num_images * test_split)])
    data_loader = DataLoader(cifar[0],
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2
                             )
    data_test_loader = DataLoader(cifar[1],
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2
                                  )
    # Setting GPU
    device = torch.device("cuda:0")
    #criterion = nn.MSELoss()    # Use it with LeNet5Dyadic
    criterion = nn.CrossEntropyLoss()    # Use it with LeNet5 & LeNet5DyadicConv
    # Optimizer initialization
    optimizer = optim.Adam(lenet.parameters(), lr=lr)
    # Checkpoint folder creation
    if '{x}_epochs_{t}'.format(x=num_epochs, t=net_type) not in os.listdir('checkpoints/'):
        os.mkdir('checkpoints/{x}_epochs_{t}'.format(x=num_epochs, t=net_type))
    # Training loop
    if train_flag:
        for e in range(1, num_epochs):
            train(net=lenet,
                  data_train_loader=data_loader,
                  criterion=criterion,
                  optimizer=optimizer,
                  losses=losses,
                  accuracies=accuracies,
                  epoch=e,
                  device=device
                  )
        torch.save(lenet.state_dict(), 'checkpoints/{x}_epochs_{t}/{x}_epochs_{t}.pth'.format(x=num_epochs, t=net_type))
        # Saving losses in a .txt file for plotting
        with open('checkpoints/{x}_epochs_{t}/loss.txt'.format(x=num_epochs, t=net_type), 'w+') as f:
            for loss in losses:
                f.write(str(loss) + '\n')
            f.close()
        # Saving parameters to reproduce the architecture
        with open('checkpoints/{x}_epochs_{t}/parameters.txt'.format(x=num_epochs, t=net_type), 'w+') as f:
            f.write('Model: ' + lenet.__name__ + '\n')
            f.write('Learning rate: ' + str(lr))
            f.write('Batch size: ' + str(batch_size))
            f.write('Number of epochs: ' + str(num_epochs))
            f.write('Train split: ' + str(train_split))
            f.write('Test split: ' + str(test_split))
            f.close()
    # Test loop
    if test_flag:
        test(net=lenet,
             checkpoint='checkpoints/{x}_epochs_{t}/{x}_epochs_{t}.pth'.format(x=num_epochs, t=net_type),
             data_loader=data_test_loader,
             criterion=criterion,
             device=device,
             batch_size=batch_size
             )
