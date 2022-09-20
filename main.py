import os
import sys
import matplotlib.pyplot as plt
from AI import Dqn
from optparse import OptionParser
import numpy as np
from environment import Environment, stop_sim, FIRST_ACTION, STATE_SIZE

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def print_summary(ep, train):
    max_len, max_wait, avg_len, avg_wait, avg_co2 = env.get_summary(save_stats=train)
    print("An epoch passed", ep)
    print("Max jam length", max_len)
    print("Max waiting time", max_wait)
    print("Average jam length", avg_len)
    print("Average waiting time", avg_wait)
    print("Average CO2 emissions", avg_co2)
    if train:
        print("Average epoch reward", np.mean(brain.temp_reward_window))
    print()


def evaluate_brain(last_check_best_brain):
    evaluate_function = sum(brain.temp_reward_window) / float(len(brain.temp_reward_window))
    if evaluate_function > last_check_best_brain or last_check_best_brain == np.inf:
        brain.save(model_name)
        print("Saving")
        return evaluate_function
    return last_check_best_brain


def plots():
    if scores and env.avg_tot_len and env.avg_tot_wait:
        plt.plot(scores)
        plt.show()
        plt.plot(env.avg_tot_len, marker='o')
        plt.show()
        plt.plot(env.avg_tot_wait, marker='o')
        plt.show()


def run(epochs=30, train=False, ai=True, event_cycle=5):
    ep = 0
    check_best_brain = np.inf
    event = 0
    while ep < epochs:
        event += 1
        next_state = env.restart()
        is_done = False
        action = FIRST_ACTION - 1
        while not is_done:
            if ai:
                action = brain.update(next_state)
                next_state, reward, is_done = env.step(action)
                if train:
                    brain.learn(next_state, reward)
                    scores.append(brain.score())
            else:
                action = (action+1) % 3
                is_done = env.set_action(action)

        if event % event_cycle == 0:
            ep += 1
            print_summary(ep, train)
            if train and ai:
                check_best_brain = evaluate_brain(check_best_brain)
                brain.temp_reward_window.clear()
    stop_sim()
    plots()
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
        default=40
    )
    options, args = parser.parse_args()
    return options


# main entry point
if __name__ == "__main__":
    arguments = get_options()
    brain = Dqn(STATE_SIZE, 3, 0.9)
    model_name = arguments.model_name
    run_with_gui = 'sumo-gui' if arguments.gui else 'sumo'
    env = Environment(run_with_gui, arguments.ai)
    brain.load(model_name)
    scores = []
    run(epochs=arguments.epochs, train=arguments.train, ai=arguments.ai, event_cycle=arguments.event)
