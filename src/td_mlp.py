from agent import Agent
from mlp_features import MlpFeatures
from cnn import CNN
import tensorflow as tf
import random as rnd
import utilities as u
from benchmarker import benchmark
import numpy as np
from datetime import datetime
import os
import sys
import chess
import time
import argparse

class TemporalDifference( Agent ):
    def __init__(self, model, state=None, session=None, session_path=None, wd=None, session_name=None):
        super()
        self.wd = wd
        self.session_name = session_name
        if self.wd is None:
            self.wd = os.getcwd()
        if self.session_name is None:
            self.session_name = 'RL'
        
        self.model = model

        self.save_path = self.wd+'/learnt/'+self.session_name+'/'

        if not os.path.exists(wd):
            os.makedirs(self.wd)
        if not os.path.exists(self.wd+'/datasets/'):
            os.makedirs(self.wd+'/datasets/')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path) 

        self.batch_size = 256
        
        self.X = model.X
        self.Y = tf.placeholder("float", shape=[None, 1])
        
        self.session = None

        self.ev = model.ev

        self.grads = tf.gradients(self.ev, self.model.trainables)

        self.optimizer = tf.train.AdamOptimizer()
        
        self.grads_s = [tf.placeholder(tf.float32, shape=tvar.get_shape()) for tvar in self.model.trainables]
        self.apply_grads = self.optimizer.apply_gradients(zip(self.grads_s, self.model.trainables))

        self.init = tf.global_variables_initializer()

        # self.must_cleanup = True

        self.session = tf.Session()
        self.session.run(self.init)
        
        self.saver = tf.train.Saver(max_to_keep=0)

        if session is not None:
            # self.must_cleanup = False
            self.session = session
        elif session_path is not None:
            self.saver.restore(self.session, session_path)


    def compute_gradients(self, root, _lambda=0.7, depth=6):
        
        # initialise the step in the trainable parameters to be passed to the optimizer
        delta_W = [np.zeros((trainable.shape)) for trainable in self.model.trainables]
        
        board = chess.Board(root)
        states = []
        scores = []

        # Play the game until the end
        while not board.is_game_over():
            state, move, score = self.choose_move(board)
            states.append(state)
            scores.append(score)
            board.push(move)
        
        #Force true reward when game is stalemate
        if board.is_fivefold_repetition() or board.is_seventyfive_moves() or board.is_stalemate():
            scores[-1] = 0

        # Outer loop of the TD_leaf(λ)
        N = len(scores)
        for t in range(N-1):
            dt = scores[t+1] - scores[t]
            discount_factor = 0
            # Inner loop
            for j in range(t, N-1):
                discount_factor += _lambda ** (j-t)
            state = np.array(u.extract_features(states[t])).reshape((1,143))
            # Compute the gradient at the current state
            grads = self.session.run(self.grads, feed_dict={self.X: state})
            # Increment total gradient for all variables
            for dW, grad in zip(delta_W, grads):
                dW += grad
        
        return delta_W
    
    def next_action(self, board):
        _, move, _ = self.choose_move(board)
        return move

    def choose_move(self, board):
        
        wins = []
        losses = []
        draws = []
        
        for move in board.legal_moves:
            board.push(move)

            features = np.array(u.extract_features(board.fen())).reshape((1,143))
            score = self.session.run(self.ev, feed_dict={self.X: features})
            
            if score == 0:
                draws.append((board.fen(), move))
            elif board.turn:
                if score == 1:
                    wins.append((board.fen(), move))
                else:
                    losses.append((board.fen(), move))
            else:
                if score == -1:
                    wins.append((board.fen(), move))
                else:
                    losses.append((board.fen(), move))
            board.pop()
            
        #Make sure we have at least one candidate move
        assert(wins or losses or draws)

        if wins:
            return (*rnd.choice(wins), 1 if board.turn else -1)
        elif draws:
            return (*rnd.choice(draws), 0)
        else:
            return (*rnd.choice(losses), -1 if board.turn else 1)


    def alphabeta(self, board, depth=12, alpha=float('-Inf'), beta=float('+Inf'), _max=True):
        if depth == 0 or board.is_game_over():
            return ([board.fen()], self.evaluate(fen=board.fen()))

        if _max:
            v = float('-Inf')
            tree = [board.fen()]
            best_subtree = None
            for move in board.legal_moves:
                board.push(move)
                (subtree, score) = self.alphabeta(depth-1, alpha, beta, False)
                board.pop()
                if score > v:
                    v = score
                    best_subtree = subtree                
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return (tree + best_subtree, v)
        else:
            v = float('Inf')
            tree = [board.fen()]
            best_subtree = None
            for move in board.legal_moves:
                board.push(move)
                (subtree, score) = self.alphabeta(depth-1, alpha, beta, True)
                board.pop()
                if score < v:
                    v = score
                    best_subtree = subtree                
                beta = min(beta, v)
                if beta <= alpha:
                    break
            return (tree + best_subtree, v)

    def train(self):
        save_file_name = self.save_path+'{}.ckpt'.format(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
        
        fens_path = self.wd + '/datasets/fen_games'
        with open(fens_path,'r') as fen_file:
            fens = fen_file.readlines()

        errors = []
    
        epoch = 0

        for idx in range(len(fens)):
            
            fen = fens[idx]

            print('about to compute gradient')

            grads = self.compute_gradients(fen)

            print('gradient {} computed'.format(idx))

            self.session.run(self.apply_grads, feed_dict={grad_: -grad 
                                                                    for grad_, grad in zip(self.grads_s, grads) })

            epoch += 1

            if not (epoch % 1000):
                self.saver.save(self.session, save_path)
                wins, loss, draws = benchmark(self, RandomAgent(), rnd.sample(fens, 100))

        print(errors)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='../data')
    parser.add_argument('-n', '--name_session', default='RL')
    parser.add_argument('-m', '--model')

    args = parser.parse_args()

    wd = args.directory
    sn = args.name_session

    if args.model == 'mlp':
        model = MlpFeatures()
    elif args.model == 'cnn':
        model = CNN()
    else:
        print('Model {} not found'.format(args.model))
        quit()
    
    model = TemporalDifference(model, wd=wd, session_name=sn)

    model.train()