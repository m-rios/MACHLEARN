from agent import Agent
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

class TdMlp( Agebt ):
    def __init__(self, state=None, session=None, session_path=None, wd=None):
        super()
        self.wd = wd
        if self.wd is None:
            self.wd = os.getcwd()
        
        if not os.path.exists(wd):
            os.makedirs(self.wd)
        if not os.path.exists(wd+'/datasets'):
            os.makedirs(self.wd+'/datasets')
        if not os.path.exists(wd+'/learnt'):
            os.makedirs(self.wd+'/learnt')    

        self.n_general = 3; self.n_piece_c = 12; self.n_square_c = 128
        self.n_input = self.n_general + self.n_piece_c + self.n_square_c
        self.n_hidden_2 = 9

        self.batch_size = 256

        self.weights = {
            'general': tf.Variable(tf.random_normal([self.n_general, self.n_general])),
            'piece_c': tf.Variable(tf.random_normal([self.n_piece_c, self.n_piece_c])),
            'square_c': tf.Variable(tf.random_normal([self.n_square_c, self.n_square_c])),
            'hidden_2': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, 1]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_input])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([1]))
        }
        
        # if state is None:
        #     self.board = chess.Board()        
        # else:
        #     self.board = chess.Board(state)
        
        self.session = None

        self.X = tf.placeholder("float", shape=[None, self.n_input])
        self.Y = tf.placeholder("float", shape=[None, 1])

        self.ev = self.nn(self.X)

        self.grads = tf.gradients(self.ev, tf.trainable_variables())

        self.optimizer = tf.train.AdamOptimizer()
        
        self.grads_s = [tf.placeholder(tf.float32, shape=tvar.get_shape()) for tvar in tf.trainable_variables()]
        self.apply_grads = self.optimizer.apply_gradients(zip(self.grads_s, tf.trainable_variables()))

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


    # def __del__(self):
    #     if self.must_cleanup:
    #         self.session.close()


    def nn(self, x):
        # Locally connected layers
        general_i = tf.gather(x,tf.convert_to_tensor(list(range(self.n_general)), dtype=tf.int32),axis=1)
        piece_i = tf.gather(x,tf.convert_to_tensor(list(range(self.n_general,self.n_general+self.n_piece_c)), dtype=tf.int32), axis=1)
        square_i = tf.gather(x,tf.convert_to_tensor(list(range(self.n_general+self.n_piece_c, self.n_general + self.n_piece_c + self.n_square_c)), dtype=tf.int32), axis=1)
        general = tf.matmul(general_i, self.weights['general'])
        piece_c = tf.matmul(piece_i, self.weights['piece_c'])
        square_c = tf.matmul(square_i, self.weights['square_c'])
        hidden_1 = tf.nn.relu(tf.add(tf.concat([general, piece_c, square_c], 1), self.biases['b1']))
        # Fully connected layer
        hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, self.weights['hidden_2']), self.biases['b2']))
        # Output layer
        out = tf.tanh(tf.add(tf.matmul(hidden_2, self.weights['out']), self.biases['out']))
        return out

    
    def compute_gradients(self, root, _lambda=0.7, depth=6):
        
        # initialise the step in the trainable parameters to be passed to the optimizer
        delta_W = [np.zeros((trainable.shape)) for trainable in tf.trainable_variables()]
        
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
        save_path = self.wd+'/learnt/mlp_{}.ckpt'.format(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
        
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
    
    wd = '../data/'

    if len(sys.argv) > 1:
        wd = sys.argv[1]
    
    model = TdMlp(wd=wd)

    model.train()

    # model.test_play()

    # a = [np.zeros(1), np.zeros(1), np.zeros(1)]
    # b = [np.ones(1) * h for h in range(1,4)]

    # for da, db in zip(a,b):
    #     da += db
    #     pass
    # pass