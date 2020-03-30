from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        if dataset == 'MNIST':
            self.conv1 = nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3)
            self.fc1 = nn.Linear(64 * 5 * 5, 1024)
            self.fc2 = nn.Linear(1024, 256)
            self.fc3 = nn.Linear(256, 10)
        elif dataset == 'STL10':
            self.conv1 = nn.Conv2d(
                in_channels=3, out_channels=6, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(
                in_channels=6, out_channels=15, kernel_size=5)
            self.fc1 = nn.Linear(15 * 21 * 21, 1024)
            self.fc2 = nn.Linear(1024, 256)
            self.fc2 = nn.Linear(256, 10)
        elif dataset == 'CIFAR10':
            self.conv1 = nn.Conv2d(
                in_channels=3, out_channels=6, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))
        X = X.view(X.size()[0], -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return X
