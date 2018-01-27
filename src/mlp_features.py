import tensorflow as tf

class MlpFeatures( object ):
    def __init__(self):
        
        self.n_general = 5; self.n_piece_c = 14; self.n_square_c = 130
        self.n_input = self.n_general + self.n_piece_c + self.n_square_c
        self.n_hidden_2 = 9
        self.out = 3

        self.X = tf.placeholder("float", shape=[None, self.n_input - 4])
        self.turn = tf.gather(self.X, [143, 144], axis = 1)
        self.X_input = tf.gather(self.X, [i for i in range(143)], axis = 1)


        self.weights = {
            'general': tf.Variable(tf.random_normal([self.n_general, self.n_general])),
            'piece_c': tf.Variable(tf.random_normal([self.n_piece_c, self.n_piece_c])),
            'square_c': tf.Variable(tf.random_normal([self.n_square_c, self.n_square_c])),
            'hidden_2': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.out]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_input])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.out]))
        }

        # Locally connected layers
        general_i = tf.gather(self.X_input,tf.convert_to_tensor(list(range(self.n_general-2)), dtype=tf.int32),axis=1)
        general_i = tf.concat([general_i,self.turn],1)

        piece_i = tf.gather(self.X_input,tf.convert_to_tensor(list(range(self.n_general - 2,self.n_general - 2 + self.n_piece_c - 2)), dtype=tf.int32), axis=1)
        piece_i = tf.concat([piece_i,self.turn],1)

        square_i = tf.gather(self.X_input,tf.convert_to_tensor(list(range(self.n_general - 2+self.n_piece_c - 2, self.n_general - 2 + self.n_piece_c - 2 + self.n_square_c - 2)), dtype=tf.int32), axis=1)
        square_i = tf.concat([square_i,self.turn],1)

        
        general = tf.matmul(general_i, self.weights['general'])
        piece_c = tf.matmul(piece_i, self.weights['piece_c'])
        square_c = tf.matmul(square_i, self.weights['square_c'])

        hidden_1 = tf.nn.tanh(tf.add(tf.concat([general, piece_c, square_c], 1), self.biases['b1']))
        # Fully connected layer
        hidden_2 = tf.nn.tanh(tf.add(tf.matmul(hidden_1, self.weights['hidden_2']), self.biases['b2']))
        # Output layer
        self.last_layer = tf.add(tf.matmul(hidden_2, self.weights['out']), self.biases['out'])
        self.class_prob = tf.nn.softmax(self.last_layer)
        self.ev = tf.argmax(self.class_prob, dimension=1)

        self.trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  
