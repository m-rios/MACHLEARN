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
        # td = SupervisedLearning(model=CNN(), wd='/Users/mario/Developer/MACHLEARN/data/')
        # td = TemporalDifference(model=CNN(), session_path='/Users/mario/Desktop/analysis/mario/rl_cnn/var.ckpt', wd='/Users/mario/Developer/MACHLEARN/data/')
        # td = TemporalDifference(model=MlpFeatures(), wd='/Users/mario/Developer/MACHLEARN/data/')
        # td = TemporalDifference(model=CNN(), session_path='/Users/mario/Desktop/analysis/mario/multiple output not finished/rl_cnn_multi/2018-01-22_01:28:55.ckpt', wd='../data')
        # td = TemporalDifference(model=MlpFeatures(), wd='../data')
        
        sl = SupervisedLearning(model=MlpBitmaps(), wd='../data', session_path='/Users/mario/Desktop/analysis/swaraj/new architecture/sl_mlp_asdf/2018-01-23_00:28:30.ckpt')
        # sl = SupervisedLearning(model=MlpBitmaps(), wd='../data')
        # improve, deprove, advantage_kept = benchmark(sl, StockAgent(), fens[100:200])
        improve, deprove, advantage_kept = benchmark(sl, RandomPlayer(), fens[100:200])

        print('improve: {}, deprove: {}, advantage_kept: {}'.format(improve, deprove, advantage_kept))
