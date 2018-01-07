from agent import Agent
import tensorflow as tf
import numpy as np

# class TdAgent( Agent ):
    
#     def __init__(self, state, knowledge=None):
#         super(TdAgent, self).__init__(state)
#         self.n_general = 3; self.n_piece_c = 12; self.n_square_c = 128
#         self.n_input = self.n_general + self.n_piece_c + self.n_square_c
#         self.n_hidden_2 = 3
#         self.weights = {
#             'general': tf.Variable(tf.random_normal([3, 3])),
#             'piece_c': tf.Variable(tf.random_normal([12, 12])),
#             'square_c': tf.Variable(tf.random_normal([128, 128])),
#             'hidden_2': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_2])),
#             'out': tf.Variable(tf.random_normal([self.n_hidden_2, 1]))
#         }
#         self.biases = {
#             'b1': tf.Variable(tf.random_normal([self.n_input])),
#             'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
#             'out': tf.Variable(tf.random_normal([1]))
#         }
#         # self.graph = self.nn(tf.placeholder("float32", [None, self.n_input]))
#     def next_action(self):
#         pass
    
#     def nn(self, x):
#         # Locally connected layers
#         general_i = tf.gather(x,tf.convert_to_tensor(list(range(self.n_general)), dtype=tf.int32),axis=1)
#         piece_i = tf.gather(x,tf.convert_to_tensor(list(range(self.n_general,self.n_general+self.n_piece_c)), dtype=tf.int32), axis=1)
#         square_i = tf.gather(x,tf.convert_to_tensor(list(range(self.n_general+self.n_piece_c, self.n_general + self.n_piece_c + self.n_square_c)), dtype=tf.int32), axis=1)
#         general = tf.matmul(general_i, self.weights['general'])
#         piece_c = tf.matmul(piece_i, self.weights['piece_c'])
#         square_c = tf.matmul(square_i, self.weights['square_c'])
#         hidden_1 = tf.nn.relu(tf.add(tf.concat([general, piece_c, square_c], 1), self.biases['b1']))
#         hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, self.weights['hidden_2']), self.biases['b2']))
#         out = tf.tanh(tf.add(tf.matmul(hidden_2, self.weights['out']), self.biases['out']))
#         return out

#     def train(self):
#         X = tf.placeholder("float", [None, n_input])
#         Y = tf.placeholder("float", [None, 1])


n_general = 3; n_piece_c = 12; n_square_c = 128
n_input = n_general + n_piece_c + n_square_c
n_hidden_2 = 3
weights = {
    'general': tf.Variable(tf.random_normal([3, 3])),
    'piece_c': tf.Variable(tf.random_normal([12, 12])),
    'square_c': tf.Variable(tf.random_normal([128, 128])),
    'hidden_2': tf.Variable(tf.random_normal([n_input, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, 1]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_input])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([1]))
}

def nn(x):
        # Locally connected layers
        general_i = tf.gather(x,tf.convert_to_tensor(list(range(n_general)), dtype=tf.int32),axis=1)
        piece_i = tf.gather(x,tf.convert_to_tensor(list(range(n_general,n_general+n_piece_c)), dtype=tf.int32), axis=1)
        square_i = tf.gather(x,tf.convert_to_tensor(list(range(n_general+n_piece_c, n_general + n_piece_c + n_square_c)), dtype=tf.int32), axis=1)
        general = tf.matmul(general_i, weights['general'])
        piece_c = tf.matmul(piece_i, weights['piece_c'])
        square_c = tf.matmul(square_i, weights['square_c'])
        hidden_1 = tf.nn.relu(tf.add(tf.concat([general, piece_c, square_c], 1), biases['b1']))
        hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, weights['hidden_2']), biases['b2']))
        out = tf.tanh(tf.add(tf.matmul(hidden_2, weights['out']), biases['out']))
        return out


if __name__ == '__main__':
    # agent = TdAgent(None)
    from utilities import extract_features
    features = extract_features('N7/5K2/8/8/8/8/8/3r3k w - - 0 1')

    # features = features + [0]*128

    # features = np.array(features).reshape(1,len(features))
    
    # x = tf.placeholder("float", [None, agent.n_input])

    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     sess.run(agent.graph, {x: features})

    X = tf.placeholder("float32", [None, n_input])
    graph = nn(X)
    init = tf.global_variables_initializer()
    f = np.array(features + [0]*128).reshape(1,143)
    with tf.Session() as sess:
        sess.run(init)
        v = sess.run(graph, {X: f})
    print(v)
