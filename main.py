import os
import sys
import matplotlib.pyplot as plt
import argparse
from src.Experiments.AIPolicy import AIPolicy
from src.Experiments.HeuristicsPolicy import HeuristicsPolicy
from src.Experiments.Simulation import RunModes
from src.Experiments.StaticPolicy import StaticPolicy


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


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
        experiment = AIPolicy(arguments, RunModes.CNN, 0.9)
    elif model_type == 'mlp':
        experiment = AIPolicy(arguments, RunModes.MLP, 0.9)
    elif model_type == 'static':
        experiment = StaticPolicy(arguments)
    elif model_type == 'mwf':
        experiment = HeuristicsPolicy(arguments, RunModes.MWF)
    elif model_type == 'lqf':
        experiment = HeuristicsPolicy(arguments, RunModes.LQF)
    else:
        raise Exception('Model type not supported!')

    scores, avg_tot_len, avg_tot_wait = experiment.run()

    if arguments.train:
        plots(scores, avg_tot_len, avg_tot_wait)
        plt.show()
