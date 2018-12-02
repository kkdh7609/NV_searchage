import models
import tensorflow as tf
import numpy as np
import csv
import random

all_data = []     # for shuffling data
data_time = []
train_data = []
train_label = []

with open('train_set_21.csv', 'r') as f:
    rd = csv.reader(f)
    for line in rd:
        all_data.append(line)

# shuffle the data
random.shuffle(all_data)

for data in all_data:
    data_time.append([data[0]])
    train_data.append([float(i) for i in data[2: -1]])
    train_label.append([float(data[-1]) - 1])

# Make arrays to numpy
train_data = np.array(train_data)
train_label = np.array(train_label)

print("Data setting is done!")

hypo = []
for i in range(len(train_data)):
    hypo.append([train_data[i][0] * 0.0779 + train_data[i][1] * 0.1324 + train_data[i][2] * 0.1300 + train_data[i][3] * 0.1194 + train_data[i][4] * 0.0908 - 1.0850])
hypo = np.array(hypo)
print(hypo.shape)
print(train_label.shape)
#print(list(hypo))
#print(list(train_label))
print(models.get_accuracy(np.rint(hypo), train_label, mode=1))

# Linear Regression is started
X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placeholder(tf.float32, shape=[None, 1])
lr = 0.001
lin = models.LinReg()
h_lin = lin.forward(X)
cost_lin = tf.reduce_mean(tf.square(h_lin - Y))
train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_lin)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        if i % 10 == 0:
            hypo, y_real = sess.run([h_lin, Y], feed_dict={X: train_data, Y: train_label})
            #print(np.rint(hypo), y_real)
            acc = models.get_accuracy(np.rint(hypo), y_real, mode=1)
            print(f"step {i} accuracy = {acc}")

        sess.run(train, feed_dict={X: train_data, Y: train_label})


# Logistic regression
logistic = models.LogReg(label_size=10)
X = tf.placeholder(tf.float32, shape=[None, 5])
Y_data = tf.placeholder(tf.int32, shape=[None])
Y = tf.one_hot(Y_data, depth=10)
h_log = logistic.forward(X)
cost_log = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=h_log)
train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_log)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_train = train_label.reshape([-1])
    for i in range(100):
        if i % 10 == 0:
            hypo, y_real = sess.run([h_log, Y], feed_dict={X: train_data, Y_data: y_train})
            #print(np.rint(hypo), y_real)
            acc = models.get_accuracy(np.rint(hypo), y_real, mode=1, one_hot=True)
            print(f"step {i} accuracy = {acc}")

        sess.run(train, feed_dict={X: train_data, Y_data: y_train})

# Neural Network ( input is regression )
nn = models.NvNN(label_size=10)
X = tf.placeholder(tf.float32, shape=[None, 5])
Y = tf.placehodler([None, 1])
h = 