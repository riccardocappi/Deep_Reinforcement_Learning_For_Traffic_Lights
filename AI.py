import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Network(nn.Module):

    def __init__(self, input_size, n_actions):
        super(Network, self).__init__()
        self.input_size = input_size
        self.n_actions = n_actions
        self.hidden_layer_nodes = 120
        self.fc1 = nn.Linear(input_size, self.hidden_layer_nodes)
        self.fc2 = nn.Linear(self.hidden_layer_nodes, n_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        actions = self.fc2(x)
        return actions


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)


class Dqn():

    def __init__(self, input_size, n_actions, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.temp_reward_window = []
        self.model = Network(input_size, n_actions)
        self.memory = ReplayMemory(5000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        # self.last_reward = 0
        self.batch_size = 16

    def select_action(self, state):
        with torch.no_grad():
            probs = F.softmax(self.model(Variable(state))*75, dim=1)
        action = probs.multinomial(num_samples=1)
        return action.item()

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
        self.optimizer.step()

    def update(self, reward, new_signal, train):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        action = self.select_action(new_state)
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000000:
            del self.reward_window[0]
        if not train:
            return action
        self.memory.push(
            (self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([reward])))
        if len(self.memory.memory) > self.batch_size:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(self.batch_size)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        # self.last_reward = reward
        self.temp_reward_window.append(reward)
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def save(self):
        torch.save({'state_dict_8': self.model.state_dict(),
                    'optimizer_8': self.optimizer.state_dict(),
                    }, 'last_brain_8.pth')

    def load(self):
        if os.path.isfile('last_brain_7.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain_7.pth')
            self.model.load_state_dict(checkpoint['state_dict_7'])
            self.optimizer.load_state_dict(checkpoint['optimizer_7'])
            print("done !")
        else:
            print("no checkpoint found...")