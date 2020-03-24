from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        if dataset == 'MNIST':
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.fc1 = nn.Linear(64 * 5 * 5, 1024)
            self.fc2 = nn.Linear(1024, 256)
            self.fc3 = nn.Linear(256, 10)
        elif dataset == 'STL10':
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 15, 5)
            self.fc1 = nn.Linear(15 * 21 * 21, 1024)
            self.fc2 = nn.Linear(1024, 256)
            self.fc2 = nn.Linear(256, 10)
        elif dataset == 'CIFAR10':
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
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
