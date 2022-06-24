import os
import sys
import optparse
import matplotlib.pyplot as plt
from AI import Dqn

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # Checks for the binary in environ vars
import traci


def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options

# Gets the phases of the simulation envirorment
def get_phases():
    logic = traci.trafficlight.getAllProgramLogics(id_tfl[0])
    p = logic[0].getPhases()
    return p

# Returns a dictionary where keys are phases indexes and values are phases states
def get_actions_dict():
    phases_dict = {}
    i = 0
    for p in phases:
        if "y" not in p.state:
            phases_dict[i] = p.state
            i += 1
    return phases_dict


def get_predicted_phase_state(index):
    if index not in actions:
        return ''
    return actions[index]


def get_yellows(from_phase, to_phase):
    yellow_phase = ''
    for i, c in enumerate(from_phase):
        if (c == 'G' or c == 'g') and (to_phase[i] == 'r'):
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


def set_phase(phase):
    traci.trafficlight.setRedYellowGreenState(id_tfl[0], phase)


# contains TraCI control loop
def run(epochs=20,train=True, ai=True):
    ep = 0
    check_best_brain = 0
    event = 0
    total_length = []
    total_waiting_time = []
    while ep < epochs:
        min_duration_ctr = min_duration
        yellow_duration_ctr = yellow_duration
        is_yellow = False
        pred_phase = ''
        traci.start([checkBinary('sumo-gui'), "-c", sim_name,"--no-step-log", "true","-W",
                     "--tripinfo-output", "tripinfo.xml", "--duration-log.disable", '--waiting-time-memory', '10000'])
        last_phase_index = 0
        set_phase(get_predicted_phase_state(last_phase_index))
        event += 1
        last_jam_length = get_detectors_jam_length()
        while traci.simulation.getMinExpectedNumber() > 0:
            if ai:
                if is_yellow:
                    yellow_duration_ctr -= 1
                    if yellow_duration_ctr == 0:
                        set_phase(pred_phase)
                        is_yellow = False
                        yellow_duration_ctr = yellow_duration
                else:
                    min_duration_ctr -= 1
                    if min_duration_ctr == 0:
                        # AI Code
                        jam_lenght = get_detectors_jam_length()
                        waiting_time = get_max_waiting_time_per_lane()
                        signal = [last_phase_index] + jam_lenght + waiting_time
                        sum_waiting_time = sum(waiting_time)
                        sum_jam_length = sum(jam_lenght)
                        total_length += jam_lenght
                        total_waiting_time += waiting_time

                        reward = (sum(last_jam_length) - sum_jam_length) + (
                                -0.3*sum_waiting_time)
                        action = brain.update(reward, signal, train)
                        scores.append(brain.score())
                        pred_phase = get_predicted_phase_state(action)
                        yellow_phase = get_yellows(get_predicted_phase_state(last_phase_index), pred_phase)
                        last_phase_index = action
                        last_jam_length = jam_lenght
                        set_phase(yellow_phase)
                        is_yellow = "y" in yellow_phase
                        min_duration_ctr = min_duration
            else:
                total_length += get_detectors_jam_length()
                total_waiting_time += get_max_waiting_time_per_lane()

            traci.simulationStep()

        if event % event_cycle == 0:
            ep += 1
            avg_waiting_time = sum(total_waiting_time) / len(total_waiting_time)
            print("An epoch passed", ep)
            print("Max jam length", max(total_length))
            print("Max waiting time", max(total_waiting_time))
            print("Average jam length", sum(total_length) / len(total_length))
            print("Average waiting time", avg_waiting_time)
            print("Memory size", len(brain.memory.memory))
            print()
            evaluate_brain = avg_waiting_time
            total_length = []
            total_waiting_time = []
            if check_best_brain > 0:
                if evaluate_brain < check_best_brain:
                    brain.save()
                    print("Saving")
                    check_best_brain = evaluate_brain
            else:
                check_best_brain = evaluate_brain

        traci.close()
        sys.stdout.flush()

    plt.plot(scores)
    plt.show()
    print("Training concluded")


# main entry point
if __name__ == "__main__":
    options = get_options()

    # check binary
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    sim_name = "./Simulation/osm_1.sumocfg"
    # traci starts sumo as a subprocess and then this script connects and runs
    traci.start([checkBinary('sumo'), "-c", sim_name, "--no-step-log", "true", "-W",
                 "--tripinfo-output", "tripinfo.xml", "--duration-log.disable"])
    id_tfl = traci.trafficlight.getIDList()
    phases = get_phases()
    actions = get_actions_dict()

    controlled_lanes_id = traci.trafficlight.getControlledLanes(id_tfl[0])
    controlled_lanes_id = list(dict.fromkeys(controlled_lanes_id))

    multilane = {
        "via_inn_fin": ("via_inn_start","via_inn_int"),
        "via_inn_fin_1": ("via_inn_start_1", "via_inn_int_1"),
        "406769345_0": ("-406769344#0_0","-406769344#2_0"),
        "406769345_1": ("-406769344#0_1", "-406769344#2_1")
    }

    traci.close()

    event_cycle = 5
    brain = Dqn(13, 3, 0.99)
    brain.load()
    scores = []
    min_duration = 10
    yellow_duration = 6
    run(train=False)
