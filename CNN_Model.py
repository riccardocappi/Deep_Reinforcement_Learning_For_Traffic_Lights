import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, input_shape, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(8, 8), stride=(4, 4))
        self.convolution2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(2, 2))
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5))
        self.fc1 = nn.Linear(in_features=self.count_neurons(input_shape), out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=number_actions)

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(self.convolution1(x))
        x = F.relu(self.convolution2(x))
        x = F.relu(self.convolution3(x))
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.convolution1(x))
        x = F.relu(self.convolution2(x))
        x = F.relu(self.convolution3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

