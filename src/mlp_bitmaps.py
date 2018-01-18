import tensorflow as tf

class MlpBitmaps( object ):
    def __init__(self):
        self.n_inputs = 64*4 
        self.n_hidden = 32 
        self.n_out = 1 

        self.weights = { 
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])), 

            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_out])) 
         } 
 
        self.biases = { 
            'hidden': tf.Variable(tf.random_normal([self.n_hidden])), 
            'out': tf.Variable(tf.random_normal([self.n_out])), 
        } 
 
        self.X = tf.placeholder("float", shape=[None, self.n_inputs])

        self.ev = self.mlp(self.X)

        self.trainables = tf.trainable_variables()
    
    def mlp(self, x):
        hidden =  tf.nn.softmax(tf.add(tf.matmul(x,self.weights['hidden']),self.biases['hidden']))
        out = tf.nn.softmax(tf.add(tf.matmul(hidden,self.weights['out']),self.biases['out']))
        ret = tf.sign(tf.subtract(out, tf.constant(0.5)))
        return ret
