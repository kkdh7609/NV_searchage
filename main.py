import models
import tensorflow as tf
import numpy as np
import csv
import random

all_data = []     # for shuffling data
data_time = []
train_data = []
train_label = []
test_data = []
test_label = []

with open('train_set211.csv', 'r') as f:
    rd = csv.reader(f)
    for line in rd:
        all_data.append(line)

with open('test_set.csv', 'r') as f:
    rd = csv.reader(f)
    for line in rd:
        test_data.append([float(i) for i in line[2:-1]])
        test_label.append([float(line[-1]) - 1])

# shuffle the data
random.shuffle(all_data)

for data in all_data:
    data_time.append([data[0]])
    train_data.append([float(i) for i in data[2: -1]])
    train_label.append([float(data[-1]) - 1])

# Make arrays to numpy
train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)

print("Data setting is done!")

"""
hypo = []
for i in range(len(train_data)):
    hypo.append([train_data[i][0] * 0.0779 + train_data[i][1] * 0.1324 + train_data[i][2] * 0.1300 + train_data[i][3] * 0.1194 + train_data[i][4] * 0.0908 - 1.0850])
hypo = np.array(hypo)
print(hypo.shape)
print(train_label.shape)
#print(list(hypo))
#print(list(train_label))
print(models.get_accuracy(np.rint(hypo), train_label, mode=1))
"""

# Neural Network ( input, output are categorical )
lr = 0.0015
nn = models.NvNN(label_size=10, one_hot=True, out_one_hot=True)
nn_train = (np.arange(train_data.max()) == train_data[..., None] - 1).astype(float)
nn_test = (np.arange(test_data.max()) == test_data[..., None] - 1).astype(float)
X_data = tf.placeholder(tf.float32, shape=[None, 4, 20])
X = tf.reshape(X_data, shape=[-1, 80])
Y_data = tf.placeholder(tf.int32, shape=[None])
Y = tf.one_hot(Y_data, depth=10)
keep_prob = tf.placeholder(tf.float32)
h_nn = nn.forward(X, keep_prob=keep_prob)
h_out = nn.forward(X, keep_prob=keep_prob)
cost_nn = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=h_nn)
train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_nn)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = train_label.reshape([-1])
    y_test = test_label.reshape([-1])
    for i in range(2000):
        if i % 10 == 0:
            hypo_test, test_y = sess.run([h_nn, Y], feed_dict={X_data: nn_test, Y_data: y_test, keep_prob: 1.0})
            act1 = models.get_accuracy(hypo_test, test_y, mode=0, one_hot=True)
            act2 = models.get_accuracy(hypo_test, test_y, mode=1, one_hot=True)
            print(f"step {i} test acc1 = {act1}, act2 = {act2}")
        sess.run(train, feed_dict={X_data: nn_train, Y_data: y_train, keep_prob: 0.8})
"""
     # For check loss
    cost_arr = sess.run(cost_nn, feed_dict={X_data: nn_train, Y_data: y_train, keep_prob: 1.0})
    with open('error.txt', 'w') as f:
        for err in cost_arr:
            f.write(str(err))
            f.write('\n')
    

# Linear Regression is started
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
lr = 0.001
lin = models.LinReg()
h_lin = lin.forward(X)
cost_lin = tf.reduce_mean(tf.square(h_lin - Y))
train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_lin)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3000):
        if i % 10 == 0:
            hypo, y_real = sess.run([h_lin, Y], feed_dict={X: train_data, Y: train_label})
            hypo_test, y_test = sess.run([h_lin, Y], feed_dict={X: test_data, Y:test_label})
            acc1 = models.get_accuracy(np.rint(hypo), y_real, mode=0)
            acc2 = models.get_accuracy(np.rint(hypo), y_real, mode=2)

            act1 = models.get_accuracy(np.rint(hypo_test), y_test, mode=0)
            act2 = models.get_accuracy(np.rint(hypo_test), y_test, mode=2)
            print(f"step {i} accuracy1 = {acc1}, acc2 = {acc2}")
            print(f"step {i} test acc1 = {act1}, act2 = {act2}")

        sess.run(train, feed_dict={X: train_data, Y: train_label})


# Logistic regression
lr = 0.001
logistic = models.LogReg(label_size=10)
X = tf.placeholder(tf.float32, shape=[None, 4])
Y_data = tf.placeholder(tf.int32, shape=[None])
Y = tf.one_hot(Y_data, depth=10)
h_log = logistic.forward(X)
cost_log = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=h_log)
train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_log)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = train_label.reshape([-1])
    y_test = test_label.reshape([-1])
    for i in range(3000):
        if i % 10 == 0:
            hypo, y_real = sess.run([h_log, Y], feed_dict={X: train_data, Y_data: y_train})
            hypo_test, test_y = sess.run([h_log, Y], feed_dict={X: test_data, Y_data: y_test})
            acc1 = models.get_accuracy(hypo, y_real, mode=0, one_hot=True)
            acc2 = models.get_accuracy(hypo, y_real, mode=1, one_hot=True)

            act1 = models.get_accuracy(hypo_test, test_y, mode=0, one_hot=True)
            act2 = models.get_accuracy(hypo_test, test_y, mode=1, one_hot=True)
            print(f"step {i} accuracy1 = {acc1}, acc2 = {acc2}")
            print(f"step {i} test acc1 = {act1}, act2 = {act2}")

        sess.run(train, feed_dict={X: train_data, Y_data: y_train})


# Neural Network ( input is regression )
nn = models.NvNN(label_size=10)
lr = 0.0015
X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)
h_nn = nn.forward(X, keep_prob=keep_prob)
cost_nn = tf.reduce_mean(tf.square(h_nn - Y))
train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_nn)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        if i % 10 == 0:
            hypo, y_real = sess.run([h_nn, Y], feed_dict={X: train_data, Y: train_label, keep_prob: 1.0})
            hypo_test, y_test = sess.run([h_nn, Y], feed_dict={X: test_data, Y: test_label, keep_prob: 1.0})
            acc1 = models.get_accuracy(np.rint(hypo), y_real, mode=0)
            acc2 = models.get_accuracy(np.rint(hypo), y_real, mode=2)

            act1 = models.get_accuracy(np.rint(hypo_test), y_test, mode=0)
            act2 = models.get_accuracy(np.rint(hypo_test), y_test, mode=2)
            print(f"step {i} accuracy1 = {acc1}, acc2 = {acc2}")
            print(f"step {i} test acc1 = {act1}, act2 = {act2}")

        sess.run(train, feed_dict={X: train_data, Y: train_label, keep_prob: 0.8})


# Neural Network ( output is categorical )
lr = 0.0015
nn = models.NvNN(label_size=10, out_one_hot=True)
X = tf.placeholder(tf.float32, shape=[None, 4])
Y_data = tf.placeholder(tf.int32, shape=[None])
Y = tf.one_hot(Y_data, depth=10)
keep_prob = tf.placeholder(tf.float32)
h_nn = nn.forward(X, keep_prob=keep_prob)
cost_nn = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=h_nn)
train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_nn)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = train_label.reshape([-1])
    y_test = test_label.reshape([-1])
    for i in range(5000):
        if i % 10 == 0:
            hypo, y_real = sess.run([h_nn, Y], feed_dict={X: train_data, Y_data: y_train, keep_prob: 1.0})
            hypo_test, test_y = sess.run([h_nn, Y], feed_dict={X: test_data, Y_data: y_test, keep_prob: 1.0})
            acc1 = models.get_accuracy(hypo, y_real, mode=0, one_hot=True)
            acc2 = models.get_accuracy(hypo, y_real, mode=1, one_hot=True)

            act1 = models.get_accuracy(hypo_test, test_y, mode=0, one_hot=True)
            act2 = models.get_accuracy(hypo_test, test_y, mode=1, one_hot=True)
            print(f"step {i} accuracy1 = {acc1}, acc2 = {acc2}")
            print(f"step {i} test acc1 = {act1}, act2 = {act2}")

        sess.run(train, feed_dict={X: train_data, Y_data: y_train, keep_prob: 0.8})


# Neural Network ( input is categorical, output is regression )
lr = 0.0015
nn_train = (np.arange(train_data.max()) == train_data[..., None] - 1).astype(float)
nn_test = (np.arange(test_data.max()) == test_data[..., None] - 1).astype(float)
nn = models.NvNN(label_size=10, one_hot=True, out_one_hot=False)
X_data = tf.placeholder(tf.float32, shape=[None, 4, 20])
X = tf.reshape(X_data, shape=[-1, 80])
Y = tf.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)
h_nn = nn.forward(X, keep_prob=keep_prob)
cost_nn = tf.reduce_mean(tf.square(h_nn - Y))
train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_nn)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        if i % 10 == 0:
            hypo, y_real = sess.run([h_nn, Y], feed_dict={X_data: nn_train, Y: train_label, keep_prob: 1.0})
            hypo_test, y_test = sess.run([h_nn, Y], feed_dict={X_data: nn_test, Y: test_label, keep_prob: 1.0})
            acc1 = models.get_accuracy(np.rint(hypo), y_real, mode=0)
            acc2 = models.get_accuracy(np.rint(hypo), y_real, mode=2)

            act1 = models.get_accuracy(np.rint(hypo_test), y_test, mode=0)
            act2 = models.get_accuracy(np.rint(hypo_test), y_test, mode=2)
            print(f"step {i} accuracy1 = {acc1}, acc2 = {acc2}")
            print(f"step {i} test acc1 = {act1}, act2 = {act2}")

        sess.run(train, feed_dict={X_data: nn_train, Y: train_label, keep_prob: 0.8})
"""
# RNN
nv_rnn = models.NvRNN()
rnn_train = (np.arange(train_data.max()) == train_data[..., None] - 1).astype(float)
X1 = tf.placeholder(tf.float32, shape=[None, 21])
X2 = tf.placeholder(tf.float32, shape=[None, 21])
X3 = tf.placeholder(tf.float32, shape=[None, 21])
X4 = tf.placeholder(tf.float32, shape=[None, 21])
X5 = tf.placeholder(tf.float32, shape=[None, 21])
Y_data = tf.placeholder(tf.int32, shape=[None])
Y = tf.one_hot(Y_data, depth=10)
keep_prob = tf.placeholder(tf.float32)
h_rnn = nv_rnn.forward(X1, X2, X3, X4, X5, keep_prob=keep_prob)
cost_rnn = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=h_rnn)
train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_rnn)

in_x1 = rnn_train[:, 0]
in_x2 = rnn_train[:, 1]
in_x3 = rnn_train[:, 2]
in_x4 = rnn_train[:, 3]
in_x5 = rnn_train[:, 4]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = train_label.reshape([-1])
    for i in range(10000):
        if i % 10:
            hypo, y_real = sess.run([h_rnn, Y], feed_dict={X1: in_x1, X2: in_x2,
                                                           X3: in_x3, X4: in_x4,
                                                           X5: in_x5, Y_data: y_train,
                                                           keep_prob: 1.0})
            acc = models.get_accuracy(hypo, y_real, mode=1, one_hot=True)
            print(f"step {i} accuracy = {acc}")
        sess.run(train, feed_dict={X1: in_x1, X2: in_x2, X3: in_x3, X4: in_x4, X5: in_x5, Y_data: y_train,
                                   keep_prob: 0.8})
