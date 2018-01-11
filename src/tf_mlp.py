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
    

    ret = tf.case({
                tf.reshape(tf.less(out, tf.convert_to_tensor(0.5)), []): lambda: tf.convert_to_tensor(-1,dtype=tf.float32),
                tf.reshape(tf.greater(out, tf.convert_to_tensor(0.5)), []): lambda: tf.convert_to_tensor(1,dtype=tf.float32)},
                default=lambda: out, exclusive=True)

    return ret


X = tf.placeholder("float", [None, n_inputs])
Y = tf.placeholder("float", [None, n_out])

ev = mlp(X)

loss_op = tf.losses.absolute_difference(labels=Y, predictions=ev)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
train_op = optimizer.minimize(loss_op)

init = tf.initialize_all_variables()

with open('../data/fen_games') as f:
    data = f.readlines()

with open('../data/labels') as f:
    labels = f.readlines()


errors = []

with tf.Session() as session:
    session.run(init)

    for epoch in range(10):
        
        # Get training data
        batchX_tensors = []
        batchY_tensors = []
        for point in r.sample(range(len(data)),batch_size):
            batchX_tensors.append(np.array(u.fromFen(data[point],figure='b')))
            batchY_tensors.append(np.array(int(labels[point])))
        
        # batchX = tf.train.batch(batchX_tensors, batch_size)
        # batchY = tf.train.batch(batchY_tensors, batch_size)

            batchX = batchX_tensors
            batchY = batchY_tensors

        _, cost = session.run([train_op, loss_op], feed_dict={X: batchX,
                                                            Y: batchY})
        errors.append(cost)

print(errors)                                                                
        