import tensorflow as tf
import random as r

n_inputs = 64*4
n_hidden = 32
n_out = 1

batch_size = 256

weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_out]))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_out])),
}

def mlp(x):
    hidden =  tf.nn.softmax(tf.add(tf.matmul(x,weights['hidden']),biases['hidden']))
    out = tf.nn.softmax(tf.add(tf.matmul(hidden,weights['out']),biases['out']))
    if out < 0.5:
        return -1
    if out > 0.5:
        return 1
    return 0

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

    for epoch in range(1000):
        
        # Get training data
        batchX = []
        batchY = []
        for _ in range(batch_size):
            batchX.append(data[r.randint(0, len(data)-1)])
            batchY.append(labels[r.randint(0, len(labels)-1)])
        
        _, cost = session.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
        errors.append(cost)

print(errors)                                                                
        