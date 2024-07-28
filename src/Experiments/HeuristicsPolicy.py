import numpy as np
from src.Experiments.Simulation import Simulation
from src.environment import stop_sim
from src.Experiments.Simulation import RunModes


class HeuristicsPolicy(Simulation):
    def __init__(self, args, mode, sim_path="./Simulation/osm_1.sumocfg", concat=True):
        super().__init__(args)
        self.is_mwf = mode == RunModes.MWF
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
            while not is_done:
                if self.is_mwf:
                    signal = np.array(env.get_max_waiting_time_per_lane())
                else:
                    signal = np.array(env.get_detectors_jam_length())
                action = np.argmax(signal.dot(env.lane_phase_matrix))
                is_done = env.set_action(action)
            if event % self.event_cycle == 0:
                ep += 1
                self.reset_event(env, ep)
                print()
        stop_sim()
        return self.scores, self.avg_tot_len, self.avg_tot_wait
