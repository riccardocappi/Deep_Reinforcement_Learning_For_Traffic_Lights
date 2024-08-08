from src.Experiments.Simulation import Simulation
from src.environment import stop_sim
from src.Experiments.Simulation import RunModes

class ActuatedPolicy(Simulation):
    def __init__(self, args, run_mode, sim_path="./Simulation/osm_1.sumocfg", concat=True):
        super().__init__(args)
        self.sim_path = sim_path
        self.concat = concat
        assert run_mode == RunModes.STATIC or run_mode == RunModes.ACTUATED, "Invalid model type"
        self.run_mode = run_mode

    # override
    def run(self):
        ep = 0
        event = 0
        env = self.create_env(self.sim_path, self.concat)
        while ep < self.epochs:
            event += 1
            _ = env.restart(state_as_matrix=False)
            program_id = "0" if self.run_mode == RunModes.ACTUATED else "1"
            env.actuated_control(program_id)
            if event % self.event_cycle == 0:
                ep += 1
                self.reset_event(env, ep)
                print()
        stop_sim()
        return self.scores, self.avg_tot_len, self.avg_tot_wait

