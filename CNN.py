import torch
import torch.nn as nn
from torch.autograd import Variable
from environment import FRAMES_TO_SAVE


class CNN(nn.Module):
    def __init__(self, input_shape, number_actions):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=FRAMES_TO_SAVE + 1, out_channels=16, kernel_size=(2, 10), stride=(2, 1)),
            nn.LeakyReLU()
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1,4), stride=(1, 2)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size = (1,2)),
            nn.Dropout(0.2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.count_neurons(input_shape), out_features=256),
            nn.LeakyReLU()
        )
        self.fc2 = nn.Linear(in_features=256, out_features=number_actions)

    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        return x.data.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
