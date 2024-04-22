import numpy as np

from src.AI import Dqn
from src.Architectures.CNN import CNN
from src.Architectures.MLP import Network
from src.Experiments.Simulation import Simulation
from src.environment import stop_sim
from src.Experiments.Simulation import RunModes

# from torchsummary import summary


def load_model(gamma, model_name, model, target_model):
    brain = Dqn(gamma, model, target_model)
    brain.load(model_name)
    return brain


class AIPolicy(Simulation):
    def __init__(self, args, mode, gamma, sim_path="./Simulation/osm_1.sumocfg", concat=True):
        super().__init__(args)
        self.mode = mode
        self.gamma = gamma
        self.brain = None
        self.check_best_brain = np.inf
        self.sim_path = sim_path
        self.concat = concat

    # override
    def run(self):
        ep = 0
        event = 0
        env = self.create_env(self.sim_path, self.concat)
        while ep < self.epochs:
            event += 1
            next_state = env.restart(state_as_matrix=self.state_as_matrix)
            is_done = False
            while not is_done:
                action = self.brain.update(next_state)
                next_state, reward, is_done = env.step(action, state_as_matrix=self.state_as_matrix)
                if self.train:
                    self.brain.learn(next_state, reward, is_done)
            if event % self.event_cycle == 0:
                ep += 1
                self.reset_event(env, ep)
        stop_sim()
        return self.scores, self.avg_tot_len, self.avg_tot_wait

    # override
    def create_env(self, sim_path="./Simulation/osm_1.sumocfg", concat=True):
        env = super().create_env(sim_path, concat)
        n_actions = len(env.actions)
        model = CNN(env.frames_stack.shape, n_actions) if self.mode == RunModes.CNN else Network(env.state_size, 3)
        target_model = CNN(env.frames_stack.shape, n_actions) if self.mode == RunModes.CNN else Network(env.state_size, 3)
        if not self.train:
            model.eval()
        self.brain = load_model(self.gamma, self.model_name, model,
                                target_model)
        return env

    # override
    def reset_event(self, env, ep):
        super().reset_event(env, ep)
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


