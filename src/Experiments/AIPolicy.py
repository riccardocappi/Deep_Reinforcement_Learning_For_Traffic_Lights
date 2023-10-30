import numpy as np

from src.AI import Dqn
from src.Architectures.CNN import CNN
from src.Architectures.MLP import Network
from src.Experiments.Policy import Policy
from src.Experiments.Simulation import RunModes


def load_model(gamma, model_name, model):
    brain = Dqn(gamma, model)
    brain.load(model_name)
    return brain


class AIPolicy(Policy):
    def __init__(self, args, mode, gamma):
        super().__init__(args)
        self.mode = mode
        self.gamma = gamma
        self.brain = None
        self.check_best_brain = np.inf

    # override
    def run_simulation(self, env):
        next_state = env.restart(state_as_matrix=self.state_as_matrix)
        is_done = False
        while not is_done:
            action = self.brain.update(next_state)
            next_state, reward, is_done = env.step(action, state_as_matrix=self.state_as_matrix)
            if self.train:
                self.brain.learn(next_state, reward)

    # override
    def on_create_env(self):
        env = self.create_env()
        model = CNN(env.frames_stack.shape, 3) if self.mode == RunModes.CNN else Network(env.state_size, 3)
        if not self.train:
            model.eval()
        self.brain = load_model(self.gamma, self.model_name, model)
        return env

    # override
    def on_event_reset(self, env, ep):
        self.reset_event(env, ep)
        if self.train:
            avg_epoch_rewards = np.mean(self.brain.temp_reward_window)
            self.scores.append(avg_epoch_rewards)
            print("Average epoch reward: ", avg_epoch_rewards)
            if self.save_model:
                self.__evaluate_brain()
            self.brain.temp_reward_window.clear()
        print()

    def __evaluate_brain(self):
        evaluate_function = np.mean(self.brain.temp_reward_window)
        if evaluate_function > self.check_best_brain or self.check_best_brain == np.inf:
            self.brain.save(self.model_name)
            print("Saving")
            self.check_best_brain = evaluate_function


