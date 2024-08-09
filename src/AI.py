import random
import os
import torch
import torch.nn.functional as F
import torch.optim as optim


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
        return map(lambda x: torch.stack(x, 0), samples)


class Dqn:

    def __init__(self, gamma, model, target_model):
        self.gamma = gamma
        self.temp_reward_window = []
        self.model = model
        self.memory = ReplayMemory(5000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = None
        self.last_action = 0
        self.batch_size = 32
        self.target_model = target_model
        self.TAU = 1e-3

    def select_action(self, state):
        state = torch.unsqueeze(state, dim=0)
        with torch.no_grad():
            probs = F.softmax(self.model(state)*75, dim=1)
        action = probs.multinomial(num_samples=1)
        return action.item()

    def learn_from_batches(self, batch_state, batch_next_state, batch_reward,
                           batch_action, batch_is_done):
        batch_reward = batch_reward.view(self.batch_size,)
        batch_is_done = batch_is_done.view(self.batch_size, )

        outputs = self.model(batch_state).gather(1, batch_action).squeeze(1)
        with torch.no_grad():
            next_outputs = self.target_model(batch_next_state).detach().max(1)[0]
        target = ((self.gamma * next_outputs) * (1 - batch_is_done)) + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
        self.optimizer.step()
        # Updates target model
        self.soft_update(self.model, self.target_model)

    def soft_update(self, model, target_model):
        for target_param, param in zip(target_model.parameters(),
                                           model.parameters()):
            target_param.data.copy_(self.TAU*param.data + (1-self.TAU)*target_param.data)

    def learn(self, new_signal, reward, is_done):
        new_state = torch.tensor(new_signal).float()
        self.memory.push(
            (self.last_state, new_state,
             torch.LongTensor([int(self.last_action)]),
             torch.Tensor([reward]),
             torch.Tensor([int(is_done)])))

        if len(self.memory.memory) > self.batch_size:
            batch_state, batch_next_state, batch_action, batch_reward, batch_is_done = self.memory.sample(self.batch_size)
            self.learn_from_batches(batch_state, batch_next_state,
                                    batch_reward,
                                    batch_action,
                                    batch_is_done)
        self.temp_reward_window.append(reward)

    def update(self, new_signal):
        new_state = torch.tensor(new_signal).float()
        action = self.select_action(new_state)
        self.last_action = action
        self.last_state = new_state
        return action

    def save(self, model_name):
        torch.save({'state_dict_policy': self.model.state_dict(),
                    'state_dict_target': self.target_model.state_dict(),
                    'optimizer_1': self.optimizer.state_dict(),
                    }, './'+model_name)

    def load(self, model_name):
        if os.path.isfile('./Trained Models/'+model_name):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('./Trained Models/' + model_name, map_location=torch.device('cpu'), weights_only=True)
            self.model.load_state_dict(checkpoint['state_dict_policy'])
            self.target_model.load_state_dict(checkpoint['state_dict_target'])
            self.optimizer.load_state_dict(checkpoint['optimizer_1'])
            print("done !")
        else:
            print("no checkpoint found...")
