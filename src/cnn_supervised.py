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
        
        self.batch_size = 10

        # Convolutional Layer 1 
        self.filter_size1 = 2 
        self.num_filters1 = 16

        # Convolutional Layer 2 
        self.filter_size2 = 2 
        self.num_filters2 = 32

        # fully connected layer 
        self.fc = 128

        # number of channels. Since input is 64*4
        self.num_channels = 4

        # dimensions of the input 
        self.input = 64*4

        self.out = 1

        self.X = tf.placeholder("float", shape=[None, self.input])
        self.X_input = tf.reshape(self.X, [-1, 8, 8, self.num_channels])

        self.Y = tf.placeholder("float", shape=[None, self.out])

        self.layer_conv1 = self.new_conv_layer(input=self.X_input , num_input_channels= self.num_channels, 
            filter_size=self.filter_size1, num_filters= self.num_filters1, use_pooling=True)
        
        self.layer_conv2 = self.new_conv_layer(input=self.layer_conv1 , num_input_channels= self.num_filters1, 
            filter_size=self.filter_size2, num_filters= self.num_filters2, use_pooling=True)

        self.layer_flat, self.num_features = self.flatten_layer(self.layer_conv2)
        self.layer_fc1 = self.new_fc_layer(input=self.layer_flat, num_inputs=self.num_features, num_outputs=self.fc, use_relu=True)
        self.layer_fc2 = self.new_fc_layer(input=self.layer_fc1, num_inputs=self.fc, num_outputs=1,use_relu=False)
        self.ev = tf.sign(tf.subtract(tf.nn.softmax(self.layer_fc2), tf.constant(0.5)))


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


    def new_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

         # biases are fixed, so we only train weights 
    def new_biases(self, length):
        return tf.Variable(tf.constant(0.05, shape=[length]))   


    def new_conv_layer(self, input, num_input_channels, filter_size, num_filters, use_pooling=True):
        # num_input_channels = num_filters in last layer 

        #shape of filter weights for convolution 
        shape = [filter_size, filter_size, num_input_channels, num_filters]

        #create new weights with the given shape
        weights = self.new_weights(shape=shape)

        #new biases, one for each filter 
        biases = self.new_biases(length=num_filters)

        # the CNN layer, padding = "same" means the input is padded with zero's. We have strides of 1,1,1,1
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')

        # a bias value added to each filter channel
        layer += biases

        # It calculates max(x, 0) for each input value x. This adds some non-linearity.
        layer = tf.nn.relu(layer)

        #  This is 2x2 max-pooling, which means that we consider 2x2 windows and select the largest value 
        # in each window. Then we move 2 pixels to the next window.
        if use_pooling:
            layer = tf.nn.max_pool(value=layer,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        return layer


    # to flatten the layer of the fully connected part of the CNN
    def flatten_layer(self, layer):

        # Get the shape of the input layer.
        layer_shape = layer.get_shape()

        # The shape of the flattened layer should be [num_images, img_height * img_width * num_channels]
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])

        return layer_flat, num_features

    # defining the fully-connected layer 
    def new_fc_layer(self, input, num_inputs, num_outputs, use_relu=True):
        # Create new weights and biases.
        weights = self.new_weights(shape=[num_inputs, num_outputs])
        biases = self.new_biases(length=num_outputs)

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(input, weights) + biases

        # Use ReLU?
        if use_relu:
            layer = tf.nn.relu(layer)

        return layer


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
    
    wd = None

    if len(sys.argv) > 1:
        wd = sys.argv[1]
    
    model = Mlp(wd='../data')

    model.train()

    #tf.summary.scalar() #to get nice graphs and so

   # tf.summary.histogram() # maybe use for weights 

   # tf.summary.tensor() #under development 

   # tf.summary.scalar("cross-entropy", xent)
   # 9:51, 10:29.

   # https://github.com/rdcolema/tensorflow-image-classification/blob/master/cnn.ipynb - a tutorial for CNN


   #tensorboard --logdir=/Users/vashisthdalmia/Documents/GitHub/MACHLEARN/data/summary
