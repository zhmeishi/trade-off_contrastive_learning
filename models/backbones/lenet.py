import torch
import torch.nn as nn
from torch.autograd import Function


class LeNet(nn.Module):
    def __init__(self, num_classes=10, output_dim = 256):
        super(LeNet, self).__init__()

        self.output_dim = output_dim
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU(True)
        self.flatten = nn.Flatten()
        self.linear_layer = nn.Linear(128 * 4 * 4, self.output_dim)
        self.relu = nn.ReLU(True)
        self.fc = nn.Linear(self.output_dim, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        x = self.linear_layer(x)
        x = self.relu(x)
        x = self.fc(x)

        return x