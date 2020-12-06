import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def train(net, data_train_loader, criterion, optimizer, losses, accuracies, epoch, device):
    net.train()
    loss_list = []
    for i, (images, labels) in enumerate(data_train_loader):

        optimizer.zero_grad()

        output = net(images.to(device))
        loss = criterion(output, labels.to(device))
        loss_list.append(loss.detach().item())

        if i % 50 == 0:
            net.eval()
            output = net(images.to(device))
            pred = output.detach().max(1)[1]
            total_correct = pred.eq(labels.to(device).view_as(pred)).sum()
            print('Train - Epoch %d, Batch: %d, Loss: %f, Accuracy: %f' % (epoch, i, loss.detach().cuda().item(),
                                                                           total_correct / len(images)))
            losses.append(loss.detach().cuda().item())
            accuracies.append(total_correct / len(images))
            net.train()

        loss.backward()
        optimizer.step()


def test(net, checkpoint, data_loader, criterion, device, batch_size=128):
    net.load_state_dict(torch.load(checkpoint))
    net.eval()
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    dataset_length = len(data_loader) * batch_size
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        output = net(images.to(device))
        avg_loss += criterion(output, labels.to(device)).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.to(device).view_as(pred)).sum()
    plt.imshow(torchvision.utils.make_grid(images).permute(1, 2, 0))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(images))))
    print('Predicted: ', ' '.join('%5s' % classes[pred[j]] for j in range(len(images))))
    plt.show()

    avg_loss /= dataset_length/batch_size
    print('Test - Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cuda().item(), (float(total_correct) / dataset_length) * 100) + '%')
