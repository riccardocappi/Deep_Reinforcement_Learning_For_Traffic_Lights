from src.Experiments.Policy import Policy
from src.environment import FIRST_ACTION


class StaticPolicy(Policy):
    def __init__(self, args):
        super().__init__(args)

    # override
    def run_simulation(self, env):
        _ = env.restart(state_as_matrix=False)
        is_done = False
        action = FIRST_ACTION - 1
        while not is_done:
            action = (action + 1) % 3
            is_done = env.set_action(action, ai=False)
