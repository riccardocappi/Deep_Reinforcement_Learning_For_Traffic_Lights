import traci
from sumolib import checkBinary  # Checks for the binary in environ vars
import sys
import numpy as np


def get_actions_space(phases):
    phases_mapping = []
    for p in phases:
        if "y" not in p.state:
            phases_mapping.append(p.state)
    return np.array(phases_mapping)


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
FRAMES_TO_SAVE = 2


class Environment:

    def __init__(self, run_with_gui, concat_lane=None, sim_path="./Simulation/osm_1.sumocfg"):
        if concat_lane is None:
            concat_lane = {}
        self.sim_name = sim_path
        self.run_with_gui = run_with_gui
        traci.start([checkBinary('sumo'), "-c", self.sim_name, "--no-step-log", "true", "-W",
                     "--tripinfo-output", "tripinfo.xml", "--duration-log.disable"])
        self.id_tfl = traci.trafficlight.getIDList()

        ####
        self.tl = self.id_tfl[0]
        self.actions = get_actions_space(self.get_phases())
        lanes_id = traci.trafficlight.getControlledLanes(self.tl)
        self.controlled_lanes_id = list(dict.fromkeys(lanes_id))
        self.controlled_detectors_id = self.get_controlled_detectors()
        self.state_size = len(self.controlled_lanes_id) * 2 + 1

        self.multilane = concat_lane
        self.epoch_total_length = []
        self.epoch_total_waiting_time = []

        self.last_phase_index = FIRST_ACTION
        self.last_jam_length_sum = sum(self.get_detectors_jam_length())
        self.min_duration = 10
        self.yellow_duration = 5

        max_length_lane = int(
            self.get_lane_length(max(self.controlled_lanes_id, key=lambda l: self.get_lane_length(l))))

        self.frames_stack = np.zeros((FRAMES_TO_SAVE + 1, len(self.controlled_lanes_id), max_length_lane + 1))
        self.lane_phase_matrix = self.get_lane_phase_matrix()
        self.global_sim_step = 0

    def get_phases(self):
        logic = traci.trafficlight.getAllProgramLogics(self.tl)
        p = logic[0].getPhases()
        return p

    def get_predicted_phase_state(self, index):
        if index >= len(self.actions):
            raise 'not valid action'
        return self.actions[index]

    def get_controlled_detectors(self):
        ids = traci.lanearea.getIDList()
        lanes = [traci.lanearea.getLaneID(detID) for detID in ids]
        assert len(lanes) == len(set(lanes))  # Must be only one detector per lane
        lane_id_mapping = dict((k, v) for k, v in zip(lanes, ids))
        controlled_detectors = []
        for lane in self.controlled_lanes_id:
            if lane in lane_id_mapping:
                controlled_detectors.append(lane_id_mapping[lane])
        return controlled_detectors

    def get_detectors_jam_length(self):
        # ids = ["e2_4", "e2_5", "e2_1", "e2_0", "via_inn_fin", "via_inn_fin_1"]
        returned_list = []
        for i in self.controlled_detectors_id:
            returned_list.append(self.get_lane_and_detectors_values(i, lambda x: traci.lanearea.getJamLengthMeters(x)))
        return returned_list

    def get_veh_id_per_lane(self, lane):
        return self.get_lane_and_detectors_values(lane, lambda x: traci.lane.getLastStepVehicleIDs(x))

    def get_lane_length(self, lane):
        return self.get_lane_and_detectors_values(lane, lambda x: traci.lane.getLength(x))

    def get_lane_and_detectors_values(self, id, function):
        res = function(id)
        if id in self.multilane:
            for m in self.multilane[id]:
                res += function(m)
        return res

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

    def set_phase(self, phase, duration, save_frames=False):
        j = 0
        traci.trafficlight.setRedYellowGreenState(self.tl, phase)
        for i in range(int(duration)):
            traci.simulationStep()
            self.global_sim_step += 1
            if self.global_sim_step % 10 == 0:
                self.do_stats()
            if save_frames and (i >= duration - FRAMES_TO_SAVE):
                self.get_state_matrix(j)
                j += 1

    def do_stats(self):
        self.epoch_total_length.append(self.get_detectors_jam_length())
        self.epoch_total_waiting_time.append(self.get_max_waiting_time_per_lane())

    def set_action(self, action, state_as_matrix=False):
        pred_phase = self.get_predicted_phase_state(action)
        if self.last_phase_index != action:
            yellow_phase = get_yellows(self.get_predicted_phase_state(self.last_phase_index), pred_phase)
            self.last_phase_index = action
            is_yellow = "y" in yellow_phase
            if is_yellow:
                self.set_phase(yellow_phase, self.yellow_duration)
        self.set_phase(pred_phase, self.min_duration, save_frames=state_as_matrix)
        return not traci.simulation.getMinExpectedNumber() > 0

    def step(self, action, state_as_matrix=False):
        is_done = self.set_action(action, state_as_matrix)
        jam_length = self.get_detectors_jam_length()
        waiting_time = self.get_max_waiting_time_per_lane()
        if state_as_matrix:
            signal = self.frames_stack
        else:
            signal = np.array([self.last_phase_index] + jam_length + waiting_time)
        sum_waiting_time = sum(waiting_time)
        sum_jam_length = sum(jam_length)
        reward = (self.last_jam_length_sum - sum_jam_length) + (-0.4 * sum_waiting_time)
        self.last_jam_length_sum = sum_jam_length
        return signal, reward, is_done

    def actuated_control(self):
        traci.trafficlight.setProgram(self.tl, "0")
        sim_step = 0
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_step += 1
            if sim_step % 10 == 0:
                self.do_stats()

    def clear_stats(self):
        self.epoch_total_waiting_time.clear()
        self.epoch_total_length.clear()

    def restart(self, state_as_matrix=False):
        stop_sim()
        traci.start([checkBinary(self.run_with_gui), "-c", self.sim_name, "--no-step-log", "true", "-W",
                     "--tripinfo-output", "tripinfo.xml", "--duration-log.disable", '--waiting-time-memory', '2000'])
        self.last_phase_index = FIRST_ACTION
        j = self.get_detectors_jam_length()
        self.last_jam_length_sum = sum(j)
        self.global_sim_step = 0

        for i in range(FRAMES_TO_SAVE):
            self.get_state_matrix(i)

        if state_as_matrix:
            signal = self.frames_stack
        else:
            w = self.get_max_waiting_time_per_lane()
            signal = np.array([self.last_phase_index] + j + w)
        return signal

    def get_summary(self):
        avg_wait = np.mean(self.epoch_total_waiting_time)
        avg_len = np.mean(self.epoch_total_length)
        max_len = np.max(self.epoch_total_length)
        max_wait = np.max(self.epoch_total_waiting_time)
        return max_len, max_wait, avg_len, avg_wait

    def get_state_matrix(self, i):
        self.frames_stack[i][:][:] = 0.0
        if i == FRAMES_TO_SAVE - 1:
            self.frames_stack[i + 1][:][:] = 0.0
        for j, lane in enumerate(self.controlled_lanes_id):
            last_veh_list = self.get_veh_id_per_lane(lane)
            for v in last_veh_list:
                veh_lane_pos = int(traci.vehicle.getDistance(v))
                self.frames_stack[i][j][veh_lane_pos] = 1.0
                if i == FRAMES_TO_SAVE - 1:
                    cumul_wait_time = traci.vehicle.getAccumulatedWaitingTime(v)
                    norm_wait_time = cumul_wait_time / 2000.0
                    self.frames_stack[i+1][j][veh_lane_pos] = norm_wait_time

    def get_epoch_jam_len_per_lane(self):
        return tuple(zip(*self.epoch_total_length))

    def get_epoch_waiting_time_per_lane(self):
        return tuple(zip(*self.epoch_total_waiting_time))

    def get_lane_phase_matrix(self):
        # Check that each signal is associated to one single link
        links = []
        for i, link in enumerate(traci.trafficlight.getControlledLinks(self.tl)):
            if len(link) > 1:
                raise 'Each signal must be associated to only one link'
            elif len(link) > 0:
                lane = link[0][0]
                links.append((lane, i))
        # Create matrix
        lane_phase_matrix = np.zeros(shape=(len(self.controlled_lanes_id), len(self.actions)))
        lane_mapping = dict((k, v) for k, v in zip(self.controlled_lanes_id, range(len(self.controlled_lanes_id))))
        for lane, link_index in links:
            i = lane_mapping[lane]
            for j, state in enumerate(self.actions):
                if state[link_index].lower() == 'g':
                    lane_phase_matrix[i][j] = 1
        return lane_phase_matrix

