import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self, input_size, n_actions):
        super(Network, self).__init__()
        self.input_size = input_size
        self.n_actions = n_actions
        self.hidden_layer_nodes = 80
        self.fc1 = nn.Linear(input_size, self.hidden_layer_nodes)
        self.fc2 = nn.Linear(self.hidden_layer_nodes, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        actions = self.fc2(x)
        return actions
