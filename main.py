import os
import sys
import matplotlib.pyplot as plt
from AI import Dqn
from optparse import OptionParser
import numpy as np
from environment import Environment, stop_sim

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def print_summary(ep):
    print("An epoch passed", ep)
    print("Max jam length", max(env.total_length))
    print("Max waiting time", max(env.total_waiting_time))
    print("Average jam length", np.mean(env.total_length))
    print("Average waiting time", np.mean(env.total_waiting_time))
    print("Memory size", len(brain.memory.memory))
    print()


def evaluate_brain(last_check_best_brain):
    evaluate_function = sum(brain.temp_reward_window) / float(len(brain.temp_reward_window))
    if evaluate_function > last_check_best_brain or last_check_best_brain == -1.0:
        brain.save(model_name)
        print("Saving")
        return evaluate_function
    return last_check_best_brain


def run(epochs=30, train=False, ai=True, event_cycle=5):
    ep = 0
    check_best_brain = -1.0
    event = 0
    action = 0
    while ep < epochs:
        event += 1
        state, reward, is_done = env.restart()
        while not is_done:
            if ai:
                action = brain.update(reward, state, train)
                scores.append(brain.score())
            else:
                action = (action + 1) % 3
            state, reward, is_done = env.step(action)
        if event % event_cycle == 0:
            ep += 1
            print_summary(ep)
            if train and ai:
                check_best_brain = evaluate_brain(check_best_brain)
                brain.temp_reward_window.clear()
            env.clear_stats()
    stop_sim()
    plt.plot(scores)
    plt.show()
    print("Training concluded")


def get_options():
    parser = OptionParser()
    parser.add_option(
        "--model_name",
        default='brain.pth',
        help="Load saved model",
        type="string"
    )
    parser.add_option(
        "--gui",
        default=False,
        action="store_true",
        help="Run using UI"
    )
    parser.add_option(
        "--train",
        default=False,
        action="store_true"
    )
    parser.add_option(
        "--event",
        type="int",
        default=5
    )
    parser.add_option(
        "--ai",
        default=True,
        action="store_true"
    )
    parser.add_option(
        "--not_ai",
        dest="ai",
        action="store_false"
    )
    parser.add_option(
        "--epochs",
        type="int",
        default=30
    )
    options, args = parser.parse_args()
    return options


# main entry point
if __name__ == "__main__":
    arguments = get_options()
    brain = Dqn(13, 3, 0.9)
    model_name = arguments.model_name
    run_with_gui = 'sumo-gui' if arguments.gui else 'sumo'
    env = Environment(run_with_gui, arguments.ai)
    brain.load(model_name)
    scores = []
    run(epochs=arguments.epochs, train=arguments.train, ai=arguments.ai, event_cycle=arguments.event)
