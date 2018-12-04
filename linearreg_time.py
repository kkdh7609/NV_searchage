import models
import tensorflow as tf
import numpy as np
import csv
import random

train_data = []
train_label = []

for i in range(24):
    train_data.append([])
    train_label.append([])

with open('train_set211.csv', 'r') as f:
    rd = csv.reader(f)
    for line in rd:
        train_data[int(line[0])].append([float(i) for i in line[2:-1]])
        train_label[int(line[0])].append([float(line[-1])])

train_data[:] = np.array(train_data[:])
train_label[:] = np.array(train_label[:])

for i in range(24):
    train_data[i] = np.array(train_data[i])
    train_label[i] = np.array(train_label[i])

X = tf.placeholder(tf.float32, shape=[None, 4])
Y = tf.placeholder(tf.float32, shape=[None, 1])
lr = 0.001
ln = []
h_ln = []
cost_ln = []
train_ln = []
for i in range(24):
    ln.append(models.LinReg())
    h_ln.append(ln[i].forward(X))
    cost_ln.append(tf.reduce_mean(tf.square(h_ln[i] - Y)))
    train_ln.append(tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_ln[i]))

with tf.Session() as sess:
    for i in range(24):
        sess.run(tf.global_variables_initializer())
        for j in range(3000):
            sess.run(train_ln[i], feed_dict={X: train_data[i], Y: train_label[i]})

        print(f"time {i}")
        print(sess.run(ln[i].coefs()))
