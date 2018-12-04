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

with open('train_set_21.csv', 'r') as f:
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

# Neural Network ( output is categorical )
nn = models.NvNN(x_size=5, label_size=10, one_hot=True, out_one_hot=True)
nn_train = (np.arange(train_data.max()) == train_data[..., None] - 1).astype(float)
nn_test = (np.arange(test_data.max()) == test_data[..., None] - 1).astype(float)
X_data = tf.placeholder(tf.float32, shape=[None, 5, 21])
X = tf.reshape(X_data, shape=[-1, 105])
Y_data = tf.placeholder(tf.int32, shape=[None])
Y = tf.one_hot(Y_data, depth=10)
keep_prob = tf.placeholder(tf.float32)
h_nn = nn.forward(X, keep_prob=keep_prob)
h_out = nn.forward(X, keep_prob=keep_prob)
cost_nn = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=h_nn)
train = tf.train.AdamOptimizer(learning_rate=0.0015).minimize(cost_nn)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = train_label.reshape([-1])
    y_test = test_label.reshape([-1])
    for i in range(5000):
        if i % 10 == 0:
            for j in range(10):
                hypo, y_real = sess.run([h_out, Y], feed_dict={X_data: nn_train[y_train == j],
                                                               Y_data: [j]*len(nn_train[y_train == j]),
                                                               keep_prob: 1.0})
                # print(np.rint(hypo), y_real)
                acc = models.get_accuracy(hypo, y_real, mode=1, one_hot=True)
                print(f"rank {j} step {i} accuracy = {acc}")

        sess.run(train, feed_dict={X_data: nn_train, Y_data: y_train, keep_prob: 0.9})
    cost_arr = sess.run(cost_nn, feed_dict={X_data: nn_train, Y_data: y_train, keep_prob: 1.0})
