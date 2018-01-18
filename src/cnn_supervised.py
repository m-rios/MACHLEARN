import tensorflow as tf
import random as r
import utilities as u
import numpy as np
from datetime import datetime
import os
import sys

class SupervisedLearning( object ):
    def __init__(self, model, session=None, session_path=None, wd=None):
       
        self.wd = wd
        if self.wd is None:
            self.wd = os.getcwd()
        
        if not os.path.exists(wd):
            os.makedirs(self.wd)
        if not os.path.exists(wd+'/datasets'):
            os.makedirs(self.wd+'/datasets')
        if not os.path.exists(wd+'/learnt'):
            os.makedirs(self.wd+'/learnt')   
        
        self.batch_size = 10
        
        self.X = model.X

        self.Y = tf.placeholder("float", shape=[None, self.out])

        self.ev = model.ev


        # optimizer 
        self.loss_op = tf.losses.mean_squared_error(labels=self.Y, predictions=self.ev)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)
        self.train_op = self.optimizer.minimize(self.loss_op)

        # to check accuracy 
        self.correct_prediction = tf.equal(self.ev, self.Y)
        self.accuracy_test= tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        self.accuracy_train = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # summary. Need to differentiate between test and training..
        self.summary_train = tf.summary.scalar("accuracy train", self.accuracy_train)
        self.summary_test = tf.summary.scalar("accuracy test", self.accuracy_test)

        self.init = tf.global_variables_initializer()

        self.session = tf.Session()
        self.session.run(self.init)
        
        self.saver = tf.train.Saver()

        if session is not None:
            self.session = session
        elif session_path is not None:
            self.saver.restore(self.session, session_path)


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
        train_acc =[]
        test_acc = []

        epoch = 0

        #should i put a self here ?
        #merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter("/Users/vashisthdalmia/Documents/GitHub/MACHLEARN/data/summary")
        

        for e in range(100):
        # for e in range(100000):

            x_batch, y_batch = zip(*r.sample(list(zip(x_batch_train, y_batch_train)), self.batch_size))

            acc1, eval_train, _ , error_train = self.session.run([self.accuracy_train, self.ev, self.train_op, self.loss_op], feed_dict={
                                                            self.X: x_batch,
                                                            self.Y: np.array(y_batch).reshape(self.batch_size,1)
                                                           })       
            s = self.session.run(self.summary_train, feed_dict={
                                                            self.X: x_batch,
                                                            self.Y: np.array(y_batch).reshape(self.batch_size,1)
                                                           })
            writer.add_summary(s,e)

            x_batch, y_batch = zip(*r.sample(list(zip(x_batch_test, y_batch_test)), self.batch_size))           
            acc2,error_test = self.session.run([self.accuracy_test, self.loss_op], feed_dict={
                                                            self.X: x_batch,
                                                            self.Y: np.array(y_batch).reshape(self.batch_size,1)
                                                           })

            s = self.session.run(self.summary_test, feed_dict={
                                                            self.X: x_batch,
                                                            self.Y: np.array(y_batch).reshape(self.batch_size,1)
                                                           })
            writer.add_summary(s,e)

            train_acc.append(acc1)
            test_acc.append(acc2)
            train_error.append(error_train)
            test_error.append(error_test)
            epoch += 1
            
            if not (epoch % 100):
                self.saver.save(self.session, save_path)
        print(train_acc)


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

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='../data')
    parser.add_argument('-n', '--name_session', default='SL')
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

    #tf.summary.scalar() #to get nice graphs and so

   # tf.summary.histogram() # maybe use for weights 

   # tf.summary.tensor() #under development 

   # tf.summary.scalar("cross-entropy", xent)
   # 9:51, 10:29.

   # https://github.com/rdcolema/tensorflow-image-classification/blob/master/cnn.ipynb - a tutorial for CNN


   #tensorboard --logdir=/Users/vashisthdalmia/Documents/GitHub/MACHLEARN/data/summary
