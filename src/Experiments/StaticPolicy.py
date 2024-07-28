from src.Experiments.Simulation import Simulation
from src.environment import stop_sim
import traci

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
            traci.trafficlight.setProgram(env.tl, "0")
            sim_step = 0
            while not not traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                sim_step += 1
                if sim_step % 10 == 0:
                    env.do_stats()
            if event % self.event_cycle == 0:
                ep += 1
                self.reset_event(env, ep)
                print()
        stop_sim()
        return self.scores, self.avg_tot_len, self.avg_tot_wait

