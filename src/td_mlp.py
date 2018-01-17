import tensorflow as tf
import random as r
import utilities as u
import numpy as np
from datetime import datetime
import os
import sys
import chess

class TdMlp( object ):
    def __init__(self, state=None, session=None, session_path=None, wd=None):
       
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
        self.Y = tf.placeholder("float", shape=[None, self.1])

        self.ev = self.nn(self.X)

        self.grads = tf.gradients(self.ev, tf.trainable_variables())

        self.loss_op = tf.losses.mean_squared_error(labels=self.Y, predictions=self.ev)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)
        self.train_op = self.optimizer.minimize(self.loss_op)

        self.init = tf.global_variables_initializer()

        self.must_cleanup = True

        self.session = tf.Session()
        self.session.run(self.init)
        
        self.saver = tf.train.Saver()

        if session is not None:
            self.must_cleanup = False
            self.session = session
        elif session_path is not None:
            self.saver.restore(self.session, session_path)


    def __del__(self):
        if self.must_cleanup:
            self.session.close()


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

    # def loss(self, fen, search_depth=12, _lambda=0.7):
        
    #     tree, _ = self.alphabeta(chess.Board(fen))

    #     error = tf.convert_to_tensor(0);

    #     loss_op = None
        
    #     grads = tf.gradients(self.ev, tf.trainable_variables())
        
    #     def sub_body(total_l, t, N, j):
    #         total_l = tf.add(total_l, tf.pow(tf.constant(_lambda),tf.subtract(j,t)))
    #         j = tf.add(j,tf.constant(1))
    #         return [total_l, t, N, j]


    #     def main_body(tree, grads, w, t, error, N):
                            
    #         total_l = tf.convert_to_tensor(0);

    #         c = lambda total_l, t, N, j: j < N
    #         tf.while_loop(c, sub_body, [tf.constant(0), t,tf.constant(search_depth), tf.constant(t)])

    #         dt = tf.subtract(tf.convert_to_tensor(self.evaluate(fen=tree[t+1])), tf.convert_to_tensor(self.evaluate(fen=tree[t])))
            
    #         v = tf.multiply(dt,total_l)

    #         grs = self.session.run(grads, {X: u.extract_features(tree[t])})

    #         error tf.multiply(grads, v)

    #         t = tf.add(t, tf.convert_to_tensor(1))
        
    #     cond = lambda tree, grads, w, t, error, N: t < N
    #     tf.while_loop(cond, main_body, [])

    # def loss(self, X, _lambda=0.7, depth=12):
    #     _loss = 0

        
    #     c1 = lambda t, N, _loss, tree : t < N - 1

    #     def b1(t, N, _loss, tree):

    #         grs_0.eva
            
    #         _loss



    #         t += 1
    #         return [t, N, _loss, tree]

    #     tf.while_loop( c1, b1, [] )      
        
    #     return _loss


    def loss(self, root, _lambda=0.7, depth=6):
        
        tree = self.alphabeta(chess.Board(root), depth=depth)

        # initialise the step in the trainable parameters to be passed to the optimizer
        delta_W = [np.zeros((trainable.shape)) for trainable in tf.trainable_variables()]
        # Outer loop of the TD_leaf(λ)
        for t in range(len(tree))




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
        errors = []
        
        epoch = 0

        for _ in range(1000):
            
            errors = self.loss(data, search_depth, _lambda, batch_size)
                    
            std_optimizer = tf.train.GradientDescentOptimizer(1)
            loss_grd = std_optimizer.compute_gradients(errors)

            ada_optimizer = tf.train.AdadeltaOptimizer()
            train = ada_optimizer.minimize(errors, grad_loss=loss_grd)
            
            self.session.run(train)

            epoch += 1

            if not (epoch % 100):
                self.saver.save(self.session, save_path)
        print(errors)


    def evaluate(self, fen=None, figure='b'):

        if fen == None:
            fen = self.board.fen()
        features = extract_features(fen)
        
        v = self.session.run(self.ev, {X: f})
        return v


    @staticmethod
    def prepare_data(wd):
        with open(wd+'/datasets/fen_games') as f:
            data = f.readlines()

        with open(wd+'/datasets/labels') as f:
            labels = f.readlines()
        
        x = []
        y = []

        # for idx in range(len(data)):
        for idx in range(100):
            x.append(np.array(u.fromFen(data[idx], figure='b')))
            y.append(int(labels[idx]))
        
        return x, y


if __name__ == '__main__':
    
    wd = None

    if len(sys.argv) > 1:
        wd = sys.argv[1]
    
    model = Mlp(wd=wd)

    model.train()



    # n_general = 3; n_piece_c = 12; n_square_c = 128
    # n_input = n_general + n_piece_c + n_square_c
    # n_hidden_2 = 9

    # batch_size = 256

    # weights = {
    #     'general': tf.Variable(tf.random_normal([n_general, n_general])),
    #     'piece_c': tf.Variable(tf.random_normal([n_piece_c, n_piece_c])),
    #     'square_c': tf.Variable(tf.random_normal([n_square_c, n_square_c])),
    #     'hidden_2': tf.Variable(tf.random_normal([n_input, n_hidden_2])),
    #     'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
    # }
    # biases = {
    #     'b1': tf.Variable(tf.random_normal([n_input])),
    #     'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    #     'out': tf.Variable(tf.random_normal([1]))
    # }
    
    # def nn(x):
    #     general_i = tf.gather(x,tf.convert_to_tensor(list(range(n_general)), dtype=tf.int32),axis=1)
    #     piece_i = tf.gather(x,tf.convert_to_tensor(list(range(n_general,n_general+n_piece_c)), dtype=tf.int32), axis=1)
    #     square_i = tf.gather(x,tf.convert_to_tensor(list(range(n_general+n_piece_c, n_general + n_piece_c + n_square_c)), dtype=tf.int32), axis=1)
    #     general = tf.matmul(general_i, weights['general'])
    #     piece_c = tf.matmul(piece_i, weights['piece_c'])
    #     square_c = tf.matmul(square_i, weights['square_c'])
    #     hidden_1 = tf.nn.relu(tf.add(tf.concat([general, piece_c, square_c], 1), biases['b1']))
    #     # Fully connected layer
    #     hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, weights['hidden_2']), biases['b2']))
    #     # Output layer
    #     out = tf.tanh(tf.add(tf.matmul(hidden_2, weights['out']), biases['out']))
    #     return out

    # X = tf.placeholder("float", shape=[None, n_input])
    # Y = tf.placeholder("float", shape=[None, 1])

    # ev = nn(X)

    # grads = tf.gradients(ev, tf.trainable_variables())

    # init = tf.global_variables_initializer()

    # features = u.extract_features('5k2/8/K7/8/8/2N5/8/r7 b - - 0 1')

    # with tf.Session() as session:
    #     session.run(init)

    #     grs = session.run([grads], feed_dict={
    #         X: np.array(features).reshape(1,143)
    #     })

    #     print(grs)