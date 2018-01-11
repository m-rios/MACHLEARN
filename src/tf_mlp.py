import tensorflow as tf
import random as r
import utilities as u
import numpy as np

n_inputs = 64*4
n_hidden = 32
n_out = 1

batch_size = 50

weights = {
    'hidden': tf.Variable(tf.random_normal([n_inputs, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_out]))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_out])),
}

def mlp(x):
    hidden =  tf.nn.softmax(tf.add(tf.matmul(x,weights['hidden']),biases['hidden']))
    out = tf.nn.softmax(tf.add(tf.matmul(hidden,weights['out']),biases['out']))
    ret = tf.sign(tf.subtract(out, tf.constant(0.5)))
    return ret


X = tf.placeholder("float", shape=[None, n_inputs])
Y = tf.placeholder("float", shape=[None, n_out])

ev = mlp(X)

# loss_op = tf.losses.absolute_difference(labels=Y, predictions=ev)

loss_op = tf.losses.mean_squared_error(labels=Y, predictions=ev)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

errors = []

def prepare_data():
    with open('../data/fen_games') as f:
        data = f.readlines()

    with open('../data/labels') as f:
        labels = f.readlines()
    
    x = []
    y = []

    for idx in range(len(data)):
        x.append(np.array(u.fromFen(data[idx], figure='b')))
        y.append(int(labels[idx]))
    
    return x, np.array(y).reshape(len(data),1)
    # return np.array(x), np.random.randn(100,1)

x_batch, y_batch = prepare_data()


with tf.Session() as session:
    session.run(init)
    for epoch in range(100):
        cost = session.run([train_op, loss_op], feed_dict={
                                                        X: x_batch,
                                                        Y: y_batch
                                                        })
        errors.append(cost)
print(errors)




print(errors)                                                                
        