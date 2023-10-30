from src.Experiments.Simulation import Simulation
from src.environment import stop_sim
from abc import ABC, abstractmethod


class Policy(Simulation, ABC):
    def __init__(self, args):
        super().__init__(args)

    # override
    def run(self):
        ep = 0
        event = 0
        env = self.on_create_env()
        while ep < self.epochs:
            event += 1
            self.run_simulation(env)
            if event % self.event_cycle == 0:
                ep += 1
                self.on_event_reset(env, ep)
        stop_sim()
        return self.scores, self.avg_tot_len, self.avg_tot_wait

    def on_create_env(self):
        return self.create_env()

    @abstractmethod
    def run_simulation(self, env):
        pass

    def on_event_reset(self, env, ep):
        self.reset_event(env, ep)
        print()
