from src.Experiments.Simulation import Simulation
from src.environment import stop_sim
from src.environment import FIRST_ACTION


class StaticPolicy(Simulation):
    def __init__(self, args, sim_path="./Simulation/osm_1.sumocfg", concat=True):
        super().__init__(args)
        self.sim_path = sim_path
        self.concat = concat

    # override
    def run(self):
        ep = 0
        event = 0
        env = self.create_env(self.sim_path, self.concat)
        while ep < self.epochs:
            event += 1
            _ = env.restart(state_as_matrix=False)
            is_done = False
            action = FIRST_ACTION - 1
            while not is_done:
                action = (action + 1) % 3
                is_done = env.set_action(action, ai=False)
            if event % self.event_cycle == 0:
                ep += 1
                self.reset_event(env, ep)
                print()
        stop_sim()
        return self.scores, self.avg_tot_len, self.avg_tot_wait

