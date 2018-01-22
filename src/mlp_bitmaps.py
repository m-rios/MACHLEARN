import tensorflow as tf

class MlpBitmaps( object ):
    def __init__(self):
        self.n_inputs = 64*4
        self.n_hidden1 = 64
        self.n_hidden2 = 32
        self.out = 3

        self.weights = { 
            'hidden1': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden1])), 
            'hidden2': tf.Variable(tf.random_normal([self.n_hidden1, self.n_hidden2])), 
            'out': tf.Variable(tf.random_normal([self.n_hidden2, self.out])) 
         } 
 
        self.biases = { 
            'hidden1': tf.Variable(tf.random_normal([self.n_hidden1])), 
            'hidden2': tf.Variable(tf.random_normal([self.n_hidden2])), 
            'out': tf.Variable(tf.random_normal([self.out])), 
        } 
 
        self.X = tf.placeholder("float", shape=[None, self.n_inputs])

        self.last_layer = self.mlp(self.X)
        self.class_prob = tf.nn.softmax(self.last_layer)
        self.ev = tf.argmax(self.class_prob, dimension=1)

        self.trainables = tf.trainable_variables()
    
    def mlp(self, x):
        hidden1 =  tf.tanh(tf.add(tf.matmul(x,self.weights['hidden1']),self.biases['hidden1']))
        hidden2 =  tf.tanh(tf.add(tf.matmul(hidden1,self.weights['hidden2']),self.biases['hidden2']))
        out = tf.add(tf.matmul(hidden2,self.weights['out']),self.biases['out'])
        # ret = tf.sign(tf.subtract(out, tf.constant(0.5)))
        return out
