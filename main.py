import os
import sys
import matplotlib.pyplot as plt
from AI import Dqn

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # Checks for the binary in environ vars
import traci


# Gets the phases of the simulation envirorment
def get_phases():
    logic = traci.trafficlight.getAllProgramLogics(id_tfl[0])
    p = logic[0].getPhases()
    return p


# Returns a dictionary where keys are phases indexes and values are phases states
def get_actions_dict():
    phases_dict = {}
    phases_duration_dict = {}
    i = 0
    for p in phases:
        if "y" not in p.state:
            phases_dict[i] = p.state
            phases_duration_dict[p.state] = p.duration
            i += 1
    return phases_dict, phases_duration_dict


def get_predicted_phase_state(index):
    if index not in actions:
        return ''
    return actions[index]


def get_yellows(from_phase, to_phase):
    yellow_phase = ''
    for i, c in enumerate(from_phase):
        if (c == 'G' or c == 'g') and (to_phase[i] == 'r' or to_phase[i] == 'R'):
            yellow_phase += 'y'
        else:
            yellow_phase += c
    return yellow_phase


def get_detectors_jam_length():
    ids = ["e2_4", "e2_5", "e2_1", "e2_0", "via_inn_fin", "via_inn_fin_1"]
    returned_list = []
    for i in ids:
        returned_list.append(get_lane_and_detectors_values(i, lambda x: traci.lanearea.getJamLengthVehicle(x)))
    return returned_list


def get_veh_id_per_lane(lane):
    return get_lane_and_detectors_values(lane, lambda x: traci.lane.getLastStepVehicleIDs(x))


def get_lane_and_detectors_values(id, function):
    if id in multilane:
        res = function(id)
        for m in multilane[id]:
            res += function(m)
        return res
    else:
        return function(id)


def get_max_waiting_time_per_lane():
    waiting_time_list = []
    for i, lane in enumerate(controlled_lanes_id):
        last_veh_list = get_veh_id_per_lane(lane)
        max_waiting_time = 0
        if len(last_veh_list) > 0:
            veh = max(last_veh_list, key=lambda x: traci.vehicle.getAccumulatedWaitingTime(x))
            max_waiting_time = traci.vehicle.getAccumulatedWaitingTime(veh)
        waiting_time_list.append(int(max_waiting_time))
    return waiting_time_list


def set_phase(phase, duration):
    traci.trafficlight.setRedYellowGreenState(id_tfl[0], phase)
    for _ in range(int(duration)):
        traci.simulationStep()


def do_stats(tot_length, tot_waiting_time):
    global total_length
    global total_waiting_time
    total_length += tot_length
    total_waiting_time += tot_waiting_time


def print_summary(ep):
    print("An epoch passed", ep)
    print("Max jam length", max(total_length))
    print("Max waiting time", max(total_waiting_time))
    print("Average jam length", sum(total_length) / len(total_length))
    print("Average waiting time", sum(total_waiting_time) / len(total_waiting_time))
    print("Memory size", len(brain.memory.memory))
    print()


def evaluate_brain(last_check_best_brain):
    evaluate_function = abs(sum(brain.temp_reward_window) / float(len(brain.temp_reward_window)))
    if evaluate_function < last_check_best_brain or last_check_best_brain == -1.0:
        brain.save()
        print("Saving")
        return evaluate_function
    return last_check_best_brain


def run_simulation(train, ai):
    last_phase_index = 0
    pred_phase = get_predicted_phase_state(last_phase_index)
    last_jam_length = get_detectors_jam_length()
    set_phase(pred_phase, min_duration if ai else phases_duration[pred_phase])
    while traci.simulation.getMinExpectedNumber() > 0:
        # AI Code
        jam_length = get_detectors_jam_length()
        waiting_time = get_max_waiting_time_per_lane()
        if ai:
            signal = [last_phase_index] + jam_length + waiting_time
            sum_waiting_time = sum(waiting_time)
            sum_jam_length = sum(jam_length)

            reward = (sum(last_jam_length) - sum_jam_length) + (-0.3 * sum_waiting_time)
            action = brain.update(reward, signal, train)
            scores.append(brain.score())
            last_jam_length = jam_length
        else:
            action = (last_phase_index + 1) % 3

        pred_phase = get_predicted_phase_state(action)
        yellow_phase = get_yellows(get_predicted_phase_state(last_phase_index), pred_phase)
        last_phase_index = action
        is_yellow = "y" in yellow_phase
        if is_yellow:
            set_phase(yellow_phase, yellow_duration)
        do_stats(get_detectors_jam_length(), get_max_waiting_time_per_lane())
        set_phase(pred_phase, min_duration if ai else phases_duration[pred_phase])


# contains TraCI control loop
def run(epochs=35, train=True, ai=True, event_cycle=5):
    ep = 0
    check_best_brain = -1.0
    event = 0
    global total_length
    global total_waiting_time
    while ep < epochs:
        event += 1
        traci.start([checkBinary('sumo'), "-c", sim_name, "--no-step-log", "true", "-W",
                     "--tripinfo-output", "tripinfo.xml", "--duration-log.disable", '--waiting-time-memory', '10000'])
        run_simulation(train, ai)
        if event % event_cycle == 0:
            ep += 1
            print_summary(ep)
            if train and ai:
                check_best_brain = evaluate_brain(check_best_brain)
                brain.temp_reward_window.clear()
            total_waiting_time = []
            total_length = []
        traci.close()
        sys.stdout.flush()

    plt.plot(scores)
    plt.show()
    print("Training concluded")


# main entry point
if __name__ == "__main__":

    sim_name = "./Simulation/osm_1.sumocfg"
    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([checkBinary('sumo'), "-c", sim_name, "--no-step-log", "true", "-W",
                 "--tripinfo-output", "tripinfo.xml", "--duration-log.disable"])
    id_tfl = traci.trafficlight.getIDList()
    phases = get_phases()
    actions, phases_duration = get_actions_dict()

    controlled_lanes_id = traci.trafficlight.getControlledLanes(id_tfl[0])
    controlled_lanes_id = list(dict.fromkeys(controlled_lanes_id))

    multilane = {
        "via_inn_fin": ("via_inn_start", "via_inn_int"),
        "via_inn_fin_1": ("via_inn_start_1", "via_inn_int_1"),
        "406769345_0": ("-406769344#0_0", "-406769344#2_0"),
        "406769345_1": ("-406769344#0_1", "-406769344#2_1")
    }

    total_length = []
    total_waiting_time = []

    traci.close()

    brain = Dqn(13, 3, 0.9)
    brain.load()
    scores = []
    min_duration = 10
    yellow_duration = 6
    run(train=False, event_cycle=1)
