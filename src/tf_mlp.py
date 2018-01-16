import tensorflow as tf
import random as r
import utilities as u
import numpy as np
from datetime import datetime
import os
import sys

class Mlp( object ):
    def __init__(self, session=None, session_path=None, wd=None):
       
        self.wd = wd
        if self.wd is None:
            self.wd = os.getcwd()
        
        if not os.path.exists(wd):
            os.makedirs(self.wd)
        if not os.path.exists(wd+'/datasets'):
            os.makedirs(self.wd+'/datasets')
        if not os.path.exists(wd+'/learnt'):
            os.makedirs(self.wd+'/learnt')    

        self.n_inputs = 64*4
        self.n_hidden = 32
        self.n_out = 1

        self.batch_size = 10

        self.weights = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_out]))
        }

        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden])),
            'out': tf.Variable(tf.random_normal([self.n_out])),
        }

        self.X = tf.placeholder("float", shape=[None, self.n_inputs])
        self.Y = tf.placeholder("float", shape=[None, self.n_out])

        self.ev = self.mlp(self.X)

        self.loss_op = tf.losses.mean_squared_error(labels=self.Y, predictions=self.ev)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)
        self.train_op = self.optimizer.minimize(self.loss_op)

        # so that this calculates the evlauation of the MLP after the training for that step has been done 
        self.eval_after = self.mlp(self.X)


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


    def mlp(self, x):
        hidden =  tf.nn.softmax(tf.add(tf.matmul(x,self.weights['hidden']),self.biases['hidden']))
        out = tf.nn.softmax(tf.add(tf.matmul(hidden,self.weights['out']),self.biases['out']))
        ret = tf.sign(tf.subtract(out, tf.constant(0.5)))
        return ret


    def train(self):
        save_path = self.wd+'/learnt/mlp_{}.ckpt'.format(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
        errors = []

        x, y = Mlp.prepare_data(self.wd)

        split = int(0.8*len(x))
        x_batch_train = x[0: split]
        y_batch_train = y[0: split]

        x_batch_test = x[split:len(x)]
        y_batch_test = y[split:len(y)]

        train_error = []
        test_error = []

        epoch = 0

        for e in range(50):
        # for e in range(100000):

            x_batch, y_batch = zip(*r.sample(list(zip(x_batch_train, y_batch_train)), self.batch_size))

            eval_train, _ , error_train = self.session.run([self.eval_after, self.train_op, self.loss_op], feed_dict={
                                                            self.X: x_batch,
                                                            self.Y: np.array(y_batch).reshape(self.batch_size,1)
                                                           })       

            x_batch, y_batch = zip(*r.sample(list(zip(x_batch_test, y_batch_test)), self.batch_size))

            _, error_test = self.session.run([self.ev, self.loss_op], feed_dict={
                                                            self.X: x_batch,
                                                            self.Y: np.array(y_batch).reshape(self.batch_size,1)
                                                           })

            train_error.append(error_train)
            test_error.append(error_test)
            epoch += 1
            
            if not (epoch % 100):
                self.saver.save(self.session, save_path)
        print(train_error)


    def evaluate(self, fen, figure='b'):
        x = u.fromFen(fen,figure)
        return self.session.run(self.ev, feed_dict={self.X: np.array(x).reshape(1,256)})


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


def test2():
    model = Mlp(session_path='../data/model2018-01-12_19:13:54.ckpt')

    with open('../data/fen_games') as f:
        with open('../data/labels') as fl:
            label = fl.readline()
            fen = f.readline()
    print('board: {}'.format(fen))
    print('label: {}'.format(label))
    print('eval: {}'.format(model.evaluate(fen)))

def test1():
    model = Mlp()
    model.train()

if __name__ == '__main__':
    
    wd = None

    if len(sys.argv) > 1:
        wd = sys.argv[1]
    
    model = Mlp(wd='../data')

    model.train()

