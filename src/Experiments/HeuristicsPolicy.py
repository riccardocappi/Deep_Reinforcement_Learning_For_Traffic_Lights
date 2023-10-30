import numpy as np

from src.Experiments.Policy import Policy
from src.Experiments.Simulation import RunModes


class HeuristicsPolicy(Policy):
    def __init__(self, args, mode):
        super().__init__(args)
        self.is_mwf = mode == RunModes.MWF

    # override
    def run_simulation(self, env):
        _ = env.restart(state_as_matrix=False)
        is_done = False
        while not is_done:
            if self.is_mwf:
                signal = np.array(env.get_max_waiting_time_per_lane())
            else:
                signal = np.array(env.get_detectors_jam_length())
            action = np.argmax(signal.dot(env.lane_phase_matrix))
            is_done = env.set_action(action, ai=True)
