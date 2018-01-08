from agent import Agent
import tensorflow as tf
import numpy as np
from utilities import extract_features
import chess

class TdAgent( Agent ):
    
    def __init__(self, state, knowledge=None):
        super(TdAgent, self).__init__(state)
        self.n_general = 3; self.n_piece_c = 12; self.n_square_c = 128
        self.n_input = self.n_general + self.n_piece_c + self.n_square_c
        self.n_hidden_2 = 9
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
        self.board = chess.Board(state)
        self.ev = 0

    def evaluate(self, state=None):
        if state == None:
            state = self.board.fen
        features = extract_features(state)
        X = tf.placeholder("float32", [None, self.n_input])
        graph = self.nn(X)
        init = tf.global_variables_initializer()
        f = np.reshape(features,(1,143))
        with tf.Session() as sess:
            sess.run(init)
            v = sess.run(graph, {X: f})
        return v
    
    def train(self):
        asdf - asdf
        pass

    def evaluate_test(self):
        self.ev += 1
        return self.ev

    def alphabeta(self, depth=12, alpha=float('-Inf'), beta=float('+Inf'), _max=True):
        if depth == 0 or self.board.is_game_over():
            return ([self.board.fen()], self.evaluate_test())

        if _max:
            v = float('-Inf')
            tree = [self.board.fen()]
            best_subtree = None
            for move in self.board.legal_moves:
                self.board.push(move)
                (subtree, score) = self.alphabeta(depth-1, alpha, beta, False)
                self.board.pop()
                if score > v:
                    v = score
                    best_subtree = subtree                
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
            return (tree + best_subtree, v)
        else:
            v = float('Inf')
            tree = [self.board.fen()]
            best_subtree = None
            for move in self.board.legal_moves:
                self.board.push(move)
                (subtree, score) = self.alphabeta(depth-1, alpha, beta, True)
                self.board.pop()
                if score < v:
                    v = score
                    best_subtree = subtree                
                beta = min(beta, v)
                if beta <= alpha:
                    break
            return (tree + best_subtree, v)


    def next_action(self):
        pass
    
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


# n_general = 3; n_piece_c = 12; n_square_c = 128
# n_input = n_general + n_piece_c + n_square_c
# n_hidden_2 = 9
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
#         # Locally connected layers
#         general_i = tf.gather(x,tf.convert_to_tensor(list(range(n_general)), dtype=tf.int32),axis=1)
#         piece_i = tf.gather(x,tf.convert_to_tensor(list(range(n_general,n_general+n_piece_c)), dtype=tf.int32), axis=1)
#         square_i = tf.gather(x,tf.convert_to_tensor(list(range(n_general+n_piece_c, n_general + n_piece_c + n_square_c)), dtype=tf.int32), axis=1)
#         general = tf.matmul(general_i, weights['general'])
#         piece_c = tf.matmul(piece_i, weights['piece_c'])
#         square_c = tf.matmul(square_i, weights['square_c'])
#         hidden_1 = tf.nn.relu(tf.add(tf.concat([general, piece_c, square_c], 1), biases['b1']))
#         # Fully connected layer
#         hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, weights['hidden_2']), biases['b2']))
#         # Output layer
#         out = tf.tanh(tf.add(tf.matmul(hidden_2, weights['out']), biases['out']))
#         return out


# if __name__ == '__main__':
#     # agent = TdAgent(None)
#     features = extract_features('N7/5K2/8/8/8/8/8/3r3k w - - 0 1')

#     X = tf.placeholder("float32", [None, n_input])
#     graph = nn(X)
#     init = tf.global_variables_initializer()
#     f = np.reshape(features,(1,143))
#     with tf.Session() as sess:
#         sess.run(init)
#         v = sess.run(graph, {X: f})
#         print(v)


if __name__ == '__main__':

    # agent = TdAgent('N7/5K2/8/8/8/8/8/3r3k w - - 0 1')
    # for _ in range(10):
    #     print(agent.evaluate())

    agent = TdAgent('N7/5K2/8/8/8/8/8/3r3k w - - 0 1')
    print('Expanding')
    (tree, score) = agent.alphabeta(depth=3)
    print(tree)
    print(score)