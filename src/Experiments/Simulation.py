from aenum import Enum
from abc import ABC, abstractmethod
from src.environment import Environment

RunModes = Enum('Modes', ['CNN', 'MLP', 'STATIC', 'MWF', 'LQF'])


def print_summary(max_len, max_wait, avg_len, avg_wait, ep):
    print("Epoch:", ep)
    print("Max jam length:", max_len)
    print("Max waiting time:", max_wait)
    print("Average jam length:", avg_len)
    print("Average waiting time:", avg_wait)


class Simulation(ABC):
    def __init__(self, args):
        model_type = args.model_type
        self.epochs = args.epochs
        self.event_cycle = args.event
        self.save_model = args.save
        self.train = args.train
        self.model_name = args.model_name
        self.state_as_matrix = model_type == 'cnn'
        self.run_with_gui = 'sumo-gui' if args.gui else 'sumo'
        self.scores = []
        self.avg_tot_len = []
        self.avg_tot_wait = []

    def create_env(self):
        concat_lane = {
            "via_inn_fin": ("via_inn_start", "via_inn_int"),
            "via_inn_fin_1": ("via_inn_start_1", "via_inn_int_1"),
            "406769345_0": ("-406769344#0_0", "-406769344#2_0"),
            "406769345_1": ("-406769344#0_1", "-406769344#2_1")
        }
        env = Environment(self.run_with_gui, concat_lane)
        return env

    def reset_event(self, env, ep):
        max_len, max_wait, avg_len, avg_wait = env.get_summary()
        print_summary(max_len, max_wait, avg_len, avg_wait, ep)
        self.avg_tot_len.append(avg_len)
        self.avg_tot_wait.append(avg_wait)
        env.clear_stats()

    @abstractmethod
    def run(self):
        pass


