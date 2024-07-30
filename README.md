# Deep Convolutional Q-Learning for Traffic Lights Optimization
This project aims at implementing a **Reinforcement Learning** agent for the **optimization of traffic lights timing** at an intersection.
## Environment
The simulation environment is implemented using [SUMO](https://sumo.dlr.de/docs/Tools/Sumolib.html), an open source traffic simulator  that allows to model real world traffic behaviours. 
The agent interacts with the environment using the Python APIs provided by [TRACI](https://sumo.dlr.de/docs/TraCI.html). This library contains a set of useful methods to get the relevant 
simulation’s information (e.g. vehicles positions, jam lengths, current signal phase, etc.) and to modify the state of the simulation (e.g. change traffic light’s phase/timing).

## Architectures
The proposed model consists in a **Convolutional Neural Network** (CNN) trained using Deep Q-learning algorithm with experience replay.
The performance of the proposed model were compared to those of the following baseline models:
- A Multi Layer Perceptron (MLP) network;
- A simple static configuration of the traffic lights signals;
- Two models which implement the _most waiting first_ (MWF) heuristic and _longest queue first_ (LQF) heuristic.

The folder Experiments contains the implementation of such models.

## Parameters
The program can be run with different arguments:
- `--model_name` determines the model name
- `--gui` default False, determines whether to run the program with or without SUMO gui.
- `--train` default False, determines whether the model is trained or not.
- `--event` default 5, number of events which compose a training epoch.
- `--epochs` default 40, number of training epochs.
- `--model_type` default "cnn", it can be "cnn", "mlp", "mwf", "lqf", "static". Determines which model is going to be executed.
- `--save` default False, determines whether to save the trained model or not.

Examples:
```
python main.py --epochs=3 --event=1 --gui

python main.py --epochs=1 --event=1 --model_type=mwf

python main.py --model_name=prova.pth --model_type=cnn --epochs=10 --event=3 --train
```

## Colab
You can find a colab notebook of the project here: https://colab.research.google.com/drive/1QC4F4s0zEoVX8Fic5CNvGqN9gEZ1Grdd?usp=sharing
