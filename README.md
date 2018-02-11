# Solving End Games in Chess: Supervised and Reinforcement learning based approaches
## Instructions to install

The Stockfish open source chess engine is needed for the project to run. It can be downloaded from [here](https://stockfishchess.org/download/).

The project has been developed in a Unix-like environment. It has not been tested with Windows. For the project to find the stockfish binaries, they must be added to $PATH, under the name 'stockfish'.

The following **Python 3** dependencies must be installed for the project to run:
* Numpy
* python-chess
* Tensorflow

## Instructions to execute

The main files are supervised.py and reinforcement.py. They implement the training and benchmarking of the models (as part of the training itself). In order to run type:

```bash
python [reinforcement.py | supervised.py] -m [mlp | cnn] -n [run name] -d [path to data folder, by default ../data]
```
The -n option sets the name of the folder where data after training will be saved.

Output is saved as Tensorflow checkpoints and can be visualized with Tensorboard.

## Final remarks

Main code can be found in src/.
Data and matlab code used for generating the plots for the paper can be found under resources/
