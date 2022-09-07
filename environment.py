import traci
from sumolib import checkBinary  # Checks for the binary in environ vars
import sys
import numpy as np


def get_actions_dict(phases):
    phases_dict = {}
    phases_duration_dict = {}
    i = 0
    for p in phases:
        if "y" not in p.state:
            phases_dict[i] = p.state
            phases_duration_dict[p.state] = p.duration
            i += 1
    return phases_dict, phases_duration_dict


def get_yellows(from_phase, to_phase):
    yellow_phase = ''
    for i, c in enumerate(from_phase):
        if (c == 'G' or c == 'g') and (to_phase[i] == 'r' or to_phase[i] == 'R'):
            yellow_phase += 'y'
        else:
            yellow_phase += c
    return yellow_phase


def stop_sim():
    traci.close()
    sys.stdout.flush()


FIRST_ACTION = 0


class Environment:

    def __init__(self, run_with_gui, run_with_ai):
        self.sim_name = "./Simulation/osm_1.sumocfg"
        self.run_with_gui = run_with_gui
        self.run_with_ai = run_with_ai
        traci.start([checkBinary('sumo'), "-c", self.sim_name, "--no-step-log", "true", "-W",
                     "--tripinfo-output", "tripinfo.xml", "--duration-log.disable"])
        self.id_tfl = traci.trafficlight.getIDList()
        self.actions, self.phases_duration = get_actions_dict(self.get_phases())

        self.controlled_lanes_id = traci.trafficlight.getControlledLanes(self.id_tfl[0])
        self.controlled_lanes_id = list(dict.fromkeys(self.controlled_lanes_id))

        self.multilane = {
            "via_inn_fin": ("via_inn_start", "via_inn_int"),
            "via_inn_fin_1": ("via_inn_start_1", "via_inn_int_1"),
            "406769345_0": ("-406769344#0_0", "-406769344#2_0"),
            "406769345_1": ("-406769344#0_1", "-406769344#2_1")
        }
        self.epoch_total_length = []
        self.epoch_total_waiting_time = []
        self.avg_tot_wait = []
        self.avg_tot_len = []
        self.co2_emissions = []

        self.last_phase_index = FIRST_ACTION
        self.last_jam_length = self.get_detectors_jam_length()
        self.min_duration = 10
        self.yellow_duration = 6

    def get_phases(self):
        logic = traci.trafficlight.getAllProgramLogics(self.id_tfl[0])
        p = logic[0].getPhases()
        return p

    def get_predicted_phase_state(self, index):
        if index not in self.actions:
            return ''
        return self.actions[index]

    def get_detectors_jam_length(self):
        ids = ["e2_4", "e2_5", "e2_1", "e2_0", "via_inn_fin", "via_inn_fin_1"]
        returned_list = []
        for i in ids:
            returned_list.append(self.get_lane_and_detectors_values(i, lambda x: traci.lanearea.getJamLengthVehicle(x)))
        return returned_list

    def get_veh_id_per_lane(self, lane):
        return self.get_lane_and_detectors_values(lane, lambda x: traci.lane.getLastStepVehicleIDs(x))

    def get_lane_and_detectors_values(self, id, function):
        res = function(id)
        if id in self.multilane:
            for m in self.multilane[id]:
                res += function(m)
        return res

    def get_total_co2_emissions(self):
        sum_co2 = 0
        for lane in self.controlled_lanes_id:
            sum_co2 += self.get_lane_and_detectors_values(lane, lambda x: traci.lane.getCO2Emission(x))
        return sum_co2

    def get_max_waiting_time_per_lane(self):
        waiting_time_list = []
        for lane in self.controlled_lanes_id:
            last_veh_list = self.get_veh_id_per_lane(lane)
            max_waiting_time = 0
            if len(last_veh_list) > 0:
                veh = max(last_veh_list, key=lambda x: traci.vehicle.getAccumulatedWaitingTime(x))
                max_waiting_time = traci.vehicle.getAccumulatedWaitingTime(veh)
            waiting_time_list.append(int(max_waiting_time))
        return waiting_time_list

    def set_phase(self, phase, duration):
        traci.trafficlight.setRedYellowGreenState(self.id_tfl[0], phase)
        for _ in range(int(duration)):
            traci.simulationStep()

    def do_stats(self):
        self.epoch_total_length.append(self.get_detectors_jam_length(),)
        self.epoch_total_waiting_time.append(self.get_max_waiting_time_per_lane())
        self.co2_emissions.append(self.get_total_co2_emissions())

    def step(self, action):
        pred_phase = self.get_predicted_phase_state(action)
        yellow_phase = get_yellows(self.get_predicted_phase_state(self.last_phase_index), pred_phase)
        self.last_phase_index = action
        is_yellow = "y" in yellow_phase
        if is_yellow:
            self.set_phase(yellow_phase, self.yellow_duration)
        self.do_stats()
        self.set_phase(pred_phase, self.min_duration if self.run_with_ai else self.phases_duration[pred_phase])
        jam_length = self.get_detectors_jam_length()
        waiting_time = self.get_max_waiting_time_per_lane()
        signal = [self.last_phase_index] + jam_length + waiting_time
        sum_waiting_time = sum(waiting_time)
        sum_jam_length = sum(jam_length)
        reward = (sum(self.last_jam_length) - sum_jam_length) + (-0.3 * sum_waiting_time)
        self.last_jam_length = jam_length
        return signal, reward, not traci.simulation.getMinExpectedNumber() > 0

    def clear_stats(self):
        self.epoch_total_waiting_time.clear()
        self.epoch_total_length.clear()
        self.co2_emissions.clear()

    def restart(self):
        stop_sim()
        traci.start([checkBinary(self.run_with_gui), "-c", self.sim_name, "--no-step-log", "true", "-W",
                     "--tripinfo-output", "tripinfo.xml", "--duration-log.disable", '--waiting-time-memory', '10000'])
        self.last_phase_index = FIRST_ACTION
        j = self.get_detectors_jam_length()
        w = self.get_max_waiting_time_per_lane()
        self.last_jam_length = j
        return [self.last_phase_index] + j + w

    def get_summary(self, save_stats=True):
        avg_wait = np.mean(self.epoch_total_waiting_time)
        avg_len = np.mean(self.epoch_total_length)
        avg_co2 = np.mean(self.co2_emissions)
        max_len = np.max(self.epoch_total_length)
        max_wait = np.max(self.epoch_total_waiting_time)
        if save_stats:
            self.avg_tot_wait.append(avg_wait)
            self.avg_tot_len.append(avg_len)
        self.clear_stats()
        return max_len, max_wait, avg_len, avg_wait, avg_co2

    def get_epoch_jam_len_per_lane(self):
        return tuple(zip(*self.epoch_total_length))

    def get_epoch_waiting_time_per_lane(self):
        return tuple(zip(*self.epoch_total_waiting_time))
