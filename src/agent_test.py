from benchmarker import benchmark
from mlp_features import MlpFeatures
from cnn import CNN
from mlp_bitmaps import MlpBitmaps
from reinforced import TemporalDifference
from supervised import SupervisedLearning
from stock_agent import StockAgent
from random_player import RandomPlayer


with open('../data/datasets/fen_games') as f:
        fens = f.readlines()
        # td = TemporalDifference(model=CNN(), wd='/Users/mario/Developer/MACHLEARN/data/')
        
        # td = SupervisedLearning(model=CNN(), session_path='/Users/mario/Desktop/analysis/swaraj/gradient_descent/sl_cnn/2018-01-20_19:34:09.ckpt', wd='/Users/mario/Developer/MACHLEARN/data/')
        td = SupervisedLearning(model=CNN(), wd='/Users/mario/Developer/MACHLEARN/data/')
        # td = TemporalDifference(model=CNN(), session_path='/Users/mario/Desktop/analysis/mario/rl_cnn/var.ckpt', wd='/Users/mario/Developer/MACHLEARN/data/')
        # td = TemporalDifference(model=MlpFeatures(), wd='/Users/mario/Developer/MACHLEARN/data/')
        wins, losses, draws = benchmark(StockAgent(depth=1), td, fens[0:49])
        # wins, losses, draws = benchmark(td , RandomPlayer(), fens[0:50])

        print('wins: {}, losses: {}, draws: {}'.format(wins, losses, draws))