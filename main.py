import os
import sys
import matplotlib.pyplot as plt
from AI import Dqn
import argparse
import numpy as np
from environment import Environment, stop_sim, FIRST_ACTION, STATE_SIZE, MATRIX_STATE_SHAPE
from AI import Network
from CNN_Model import CNN

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def print_summary(max_len, max_wait, avg_len, avg_wait, avg_rewards, ep, train):
    print("An epoch passed", ep)
    print("Max jam length", max_len)
    print("Max waiting time", max_wait)
    print("Average jam length", avg_len)
    print("Average waiting time", avg_wait)
    if train:
        print("Average epoch reward", avg_rewards)
    print()


def evaluate_brain(model_name, brain, last_check_best_brain):
    evaluate_function = np.mean(brain.temp_reward_window)
    if evaluate_function > last_check_best_brain or last_check_best_brain == np.inf:
        brain.save(model_name)
        print("Saving")
        return evaluate_function
    return last_check_best_brain


def run(env, brain, model_name, epochs=30, train=False, ai=True, event_cycle=5,
        state_as_matrix=False, save_model=True):
    ep = 0
    check_best_brain = np.inf
    event = 0
    scores = []
    avg_tot_len = []
    avg_tot_wait = []

    while ep < epochs:
        event += 1
        next_state = env.restart(state_as_matrix=state_as_matrix)
        is_done = False
        action = FIRST_ACTION - 1
        while not is_done:
            if ai:
                action = brain.update(next_state)
                next_state, reward, is_done = env.step(action, state_as_matrix=state_as_matrix)
                if train:
                    brain.learn(next_state, reward)
            else:
                action = (action + 1) % 3
                is_done = env.set_action(action)

        if event % event_cycle == 0:
            ep += 1
            avg_epoch_rewards = 0
            if train and ai:
                avg_epoch_rewards = np.mean(brain.temp_reward_window)
                scores.append(avg_epoch_rewards)
                if save_model:
                    check_best_brain = evaluate_brain(model_name, brain, check_best_brain)
                brain.temp_reward_window.clear()

            max_len, max_wait, avg_len, avg_wait = env.get_summary()
            print_summary(max_len, max_wait, avg_len, avg_wait, avg_epoch_rewards, ep, train)
            avg_tot_len.append(avg_len)
            avg_tot_wait.append(avg_wait)
            env.clear_stats()

    stop_sim()
    print("Training concluded")
    return scores, avg_tot_len, avg_tot_wait


def plots(scores, avg_tot_len, avg_tot_wait):
    _, axes = plt.subplots(3, 1, figsize=(10, 18))
    axes = axes.flatten()
    axes[0].plot(scores, color='red', linewidth=0.5)
    axes[0].title.set_text('Avg. reward')
    axes[0].set_ylabel('Average reward')
    axes[0].set_xlabel('x')

    axes[1].plot(avg_tot_len, color='b', linewidth=0.5)
    axes[1].title.set_text('Avg. Queue length')
    axes[1].set_ylabel('Average Queue length (vehicles)')
    axes[1].set_xlabel('Epochs')

    axes[2].plot(avg_tot_wait, color='orange', linewidth=0.5)
    axes[2].title.set_text('Avg. Cumulated waiting time')
    axes[2].set_ylabel('Average Cumulated waiting time (s)')
    axes[2].set_xlabel('Epochs')


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default='cnn_brain.pth',
        help="Load saved model")
    parser.add_argument(
        "--gui",
        default=False,
        action="store_true",
        help="Run using UI"
    )
    parser.add_argument(
        "--train",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--event",
        default=5,
        type=int
    )
    parser.add_argument(
        "--ai",
        default=True,
        action="store_true"
    )
    parser.add_argument(
        "--not_ai",
        dest="ai",
        action="store_false"
    )
    parser.add_argument(
        "--epochs",
        default=40,
        type=int
    )

    parser.add_argument(
        "--model_type",
        default="cnn"
    )

    parser.add_argument(
        "--save",
        default=False,
        action="store_true"
    )

    args = parser.parse_args()
    return args


# main entry point
if __name__ == "__main__":
    arguments = get_options()
    model_type = arguments.model_type

    if model_type == 'cnn':
        model = CNN(MATRIX_STATE_SHAPE, 3)
    elif model_type == 'mlp':
        model = Network(STATE_SIZE, 3)
    else:
        raise Exception('Model type not supported!')

    state_as_matrix = model_type == 'cnn'

    brain = Dqn(0.9, model)
    model_name = arguments.model_name
    run_with_gui = 'sumo-gui' if arguments.gui else 'sumo'
    save_model = arguments.save

    env = Environment(run_with_gui, arguments.ai)
    brain.load(model_name)
    scores, avg_tot_len, avg_tot_wait = \
        run(env, brain, model_name, epochs=arguments.epochs, train=arguments.train, ai=arguments.ai,
            event_cycle=arguments.event, state_as_matrix=state_as_matrix, save_model=save_model)

    if arguments.train:
        plots(scores, avg_tot_len, avg_tot_wait)
        plt.show()
