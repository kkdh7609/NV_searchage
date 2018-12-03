import models
import numpy as np
import tensorflow as tf
import csv
import random

all_data = []
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

random.shuffle(all_data)

for data in all_data:
    train_data.append([float(i) for i in data[2: -1]])
    train_label.append([float(data[-1]) - 1])

# Make arrays to numpy
train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)

train_data = (np.arange(train_data.max()) == train_data[..., None] - 1).astype(float)
test_data = (np.arange(test_data.max()) == test_data[..., None] - 1).astype(float)

X_data = tf.placeholder(tf.float32, shape=[None, 4, 20])
X = tf.reshape(X_data, shape=[-1, 80])
Y_data = tf.placeholder(tf.int32, shape=[None])
Y = tf.one_hot(Y_data, depth=10)
lr = 0.001
keep_prob = tf.placeholder(tf.float32)

nn = models.NvNN(label_size=10, one_hot=True, out_one_hot=True)
h_nn = nn.forward(X, keep_prob=keep_prob)
cost_nn = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=h_nn)
train = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_nn)

error_index1 = []
error_index2 = []
error_index3 = []

data1 = []
label1 = []
data2 = []
label2 = []
data3 = []
label3 = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_label = train_label.reshape([-1])
    test_label = test_label.reshape([-1])
    for i in range(3000):
        print(i)
        sess.run(train, feed_dict={X_data: train_data, Y_data: train_label, keep_prob: 0.8})

    ce = sess.run(cost_nn, feed_dict={X_data: train_data, Y_data: train_label, keep_prob: 1.0})
    sorted_error = np.argsort(ce, axis=0)
    error_index1 = sorted_error[:int(int(len(ce)) / 3)]
    error_index2 = sorted_error[int(int(len(ce)) / 3):int((int(len(ce)) * 2) / 3)]
    error_index3 = sorted_error[int((int(len(ce)) * 2) / 3):]
    data1 = train_data[error_index1]
    label1 = train_label[error_index1]
    data2 = train_data[error_index2]
    label2 = train_label[error_index2]
    data3 = train_data[error_index3]
    label3 = train_label[error_index3]

nn1 = models.NvNN(label_size=10, one_hot=True, out_one_hot=True)
h_nn1 = nn1.forward(X, keep_prob=keep_prob)
cost_nn1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=h_nn1)
train1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_nn1)
nn2 = models.NvNN(label_size=10, one_hot=True, out_one_hot=True)
h_nn2 = nn2.forward(X, keep_prob=keep_prob)
cost_nn2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=h_nn2)
train2 = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_nn2)
nn3 = models.NvNN(label_size=10, one_hot=True, out_one_hot=True)
h_nn3 = nn3.forward(X, keep_prob=keep_prob)
cost_nn3 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=h_nn3)
train3 = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_nn3)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        if i % 10 == 0:
            hypo1, hypo2, hypo3 = sess.run([h_nn1, h_nn2, h_nn3], feed_dict={X_data: test_data, keep_prob: 1.0})
            y_real = sess.run(Y, feed_dict={Y_data: test_label})
            hypo = hypo1 + hypo2 + hypo3
            hypo = sess.run(tf.nn.softmax(hypo))
            print(np.argsort(hypo, axis=1)[:, -1])
            acc = models.get_accuracy(hypo, y_real, mode=1, one_hot=True)
            print(acc)
        sess.run(train1, feed_dict={X_data: data1, Y_data: label1, keep_prob: 0.8})
        sess.run(train2, feed_dict={X_data: data2, Y_data: label2, keep_prob: 0.8})
        sess.run(train3, feed_dict={X_data: data3, Y_data: label3, keep_prob: 0.8})

