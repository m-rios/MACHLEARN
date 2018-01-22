import argparse
import tensorflow as tf
import random as r
import utilities as u
import numpy as np
from datetime import datetime
import os
import sys
from benchmarker import benchmark
from random_player import RandomPlayer
from stock_agent import StockAgent
from agent import Agent
from mlp_bitmaps import MlpBitmaps
from cnn import CNN

class SupervisedLearning( Agent ):
    def __init__(self, model, session=None, session_path=None, wd=None, session_name=None):
        super()
        self.wd = wd
        self.model = model
        self.session_name = session_name
        if self.wd is None:
            self.wd = os.getcwd()
        if self.session_name is None:
            self.session_name = 'SL_MLP'
        
        self.save_path = self.wd+'/learnt/'+self.session_name+'/'

        if not os.path.exists(self.wd):
            os.makedirs(self.wd)
        if not os.path.exists(self.wd+'/datasets/'):
            os.makedirs(self.wd+'/datasets/')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.batch_size = 256

        self.X = self.model.X
        self.Y = tf.placeholder("float", shape=[None, self.model.out])

        self.Y_true_cls = tf.argmax(self.Y, dimension=1)

        self.ev = self.model.ev

       # self.loss_op = tf.losses.mean_squared_error(labels=self.Y, predictions=self.ev)
       # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)
       # self.train_op = self.optimizer.minimize(self.loss_op)

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.model.last_layer,
                                                        labels=self.Y)
        self.loss_op = tf.reduce_mean(self.cross_entropy)
        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss_op)



        # to check accuracy 
        self.correct_prediction = tf.equal(self.ev, self.Y_true_cls)
        self.accuracy_test= tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        self.accuracy_train = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # summary. Need to differentiate between test and training..
        self.summary_train = tf.summary.scalar("accuracy train", self.accuracy_train)
        self.summary_test = tf.summary.scalar("accuracy test", self.accuracy_test)
        self.summary_loss = tf.summary.scalar("loss/error", self.loss_op)

        self.init = tf.global_variables_initializer()

        self.session = tf.Session()
        self.session.run(self.init)
        
        self.saver = tf.train.Saver()

        if session is not None:
            self.session = session
        elif session_path is not None:
            self.saver.restore(self.session, session_path)


    def train(self):
        save_file_name = self.save_path+'{}.ckpt'.format(datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
        errors = []

        x, y, test_games = SupervisedLearning.prepare_data(self.wd)
        
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
        evaluation = []

        #should i put a self here ?
        #merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.save_path, filename_suffix=datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))        

        for e in range(200000):

            x_batch, y_batch = zip(*r.sample(list(zip(x_batch_train, y_batch_train)), self.batch_size))

            acc1, eval_train, _ , error_train,s = self.session.run([self.accuracy_train, self.ev, self.train_op, self.loss_op, self.summary_train], feed_dict={
                                                         self.X: x_batch,
                                                            self.Y: np.array(y_batch).reshape(self.batch_size,3)
                                                          })   
            writer.add_summary(s,e)

            x_batch, y_batch = zip(*r.sample(list(zip(x_batch_test, y_batch_test)), self.batch_size))           
            acc2,error_test,s1,s2 = self.session.run([self.accuracy_test, self.loss_op, self.summary_test, self.summary_loss], feed_dict={
                                                            self.X: x_batch,
                                                            self.Y: np.array(y_batch).reshape(self.batch_size,3)
                                                           })
            writer.add_summary(s1,e)
            writer.add_summary(s2,e)

            epoch += 1

            if not (epoch % 1000):
                self.saver.save(self.session, save_file_name)
                writer.flush()

            if not (epoch % 20000):
            # if not (epoch % 1):

                self.saver.save(self.session, save_file_name)
                wins_random, losses_random, draws_random = benchmark(self, RandomPlayer(), test_games)
                wins_stock, losses_stock, draws_stock = benchmark(self, StockAgent(depth=4), test_games)
                summary=tf.Summary()
                summary.value.add(tag='wins_random', simple_value = wins_random)
                summary.value.add(tag='losses_random', simple_value = losses_random)
                summary.value.add(tag='draws_random', simple_value = draws_random)
                summary.value.add(tag='wins_stock', simple_value = wins_stock)
                summary.value.add(tag='losses_stock', simple_value = losses_stock)
                summary.value.add(tag='draws_stock', simple_value = draws_stock)
                writer.add_summary(summary, epoch)
                writer.flush()

        


    def evaluate(self, fen, figure='b'):
        x = u.fromFen(fen,figure)
        return self.session.run(self.ev, feed_dict={self.X: np.array(x).reshape(1,256)}) - 1
    

    def next_action(self, board):
        wins = []
        losses = []
        draws = []
        
        for move in board.legal_moves:
            board.push(move)

            score = self.evaluate(board.fen())
            
            if score == 0:
                draws.append(move)
            elif board.turn:
                if score == 1:
                    wins.append(move)
                else:
                    losses.append(move)
            else:
                if score == -1:
                    wins.append(move)
                else:
                    losses.append(move)
            board.pop()
            
        #Make sure we have at least one candidate move
        assert(wins or losses or draws)

        if wins:
            return r.choice(wins)
        elif draws:
            return r.choice(draws)
        else:
            return r.choice(losses)

    @staticmethod
    def prepare_data(wd):
        with open(wd+'/datasets/fen_games') as f:
            data = f.readlines()

        with open(wd+'/datasets/labels') as f:
            labels = f.readlines()
        
        x = []
        y = []

        for idx in range(len(data)):
        # for idx in range(100):
            x.append(np.array(u.fromFen(data[idx], figure='b')))
            if int(labels[idx]) == 1:
                y.append([0,0,1])
            elif int(labels[idx]) == -1:
                y.append([1,0,0])
            else:
                y.append([0,1,0])
        
        return x, y, data[len(data)-200:len(data)]

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='../data')
    parser.add_argument('-n', '--name_session', default='RL')
    parser.add_argument('-m', '--model')

    args = parser.parse_args()

    wd = args.directory
    sn = args.name_session

    if args.model == 'mlp':
        model = MlpBitmaps()
    elif args.model == 'cnn':
        model = CNN()
    else:
        print('Model {} not found'.format(args.model))
        quit()
    
    model = SupervisedLearning(model=model, wd=wd, session_name=sn)

    model.train()