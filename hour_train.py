import models
import tensorflow as tf
import numpy as np
import csv
import random

all_data = []     # for shuffling data
train_data_arr = []
train_label_arr = []
train_date = []
test_data_arr = []
test_label_arr = []
test_date = []
train_data = []
train_label = []
test_data = []
test_label = []

for i in range(6):
    train_data.append([])
    train_label.append([])
    test_data.append([])
    test_label.append([])

with open('train_set211.csv', 'r') as f:
    rd = csv.reader(f)
    for line in rd:
        all_data.append(line)

with open('test_set.csv', 'r') as f:
    rd = csv.reader(f)
    for line in rd:
        test_date.append(int(line[0]))
        test_data_arr.append([float(i) for i in line[2:-1]])
        test_label_arr.append([float(line[-1]) - 1])

# shuffle the data
random.shuffle(all_data)

for data in all_data:
    train_date.append(int(data[0]))
    train_data_arr.append([float(i) for i in data[2: -1]])
    train_label_arr.append([float(data[-1]) - 1])

train_data_arr = np.array(train_data_arr)
train_label_arr = np.array(train_label_arr)
test_data_arr = np.array(test_data_arr)
test_label_arr = np.array(test_label_arr)

train_data_arr = (np.arange(train_data_arr.max()) == train_data_arr[..., None] - 1).astype(float)
test_data_arr = (np.arange(test_data_arr.max()) == test_data_arr[..., None] - 1).astype(float)
train_label_arr = train_label_arr.reshape([-1])
test_label_arr = test_label_arr.reshape([-1])

for i in range(len(train_date)):
    train_data[train_date[i] // 4].append(train_data_arr[i])
    train_label[train_date[i] // 4] .append(train_label_arr[i])

for i in range(len(test_date)):
    test_data[test_date[i] // 4].append(test_data_arr[i])
    test_label[test_date[i] // 4].append(test_label_arr[i])

"""
# Make arrays to numpy
train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)
"""

print(len(train_data))
print(len(train_label))
X_data = tf.placeholder(tf.float32, shape=[None, 4, 20])
X = tf.reshape(X_data, shape=[-1, 80])
Y_data = tf.placeholder(tf.int32, shape=[None])
Y = tf.one_hot(Y_data, depth=10)
keep_prob = tf.placeholder(tf.float32)
lr = 0.0015
nn = []
h_nn = []
cost_nn = []
train_nn = []
for i in range(24):
    nn.append(models.NvNN(label_size=10, one_hot=True, out_one_hot=True))
    h_nn.append(nn[i].forward(X, keep_prob=keep_prob))
    cost_nn.append(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=h_nn[i]))
    train_nn.append(tf.train.AdamOptimizer(learning_rate=lr).minimize(cost_nn[i]))

with tf.Session() as sess:
    for i in range(6):
        sess.run(tf.global_variables_initializer())
        for j in range(3000):
            if j % 10 == 0:
                hypo, y_real = sess.run([h_nn[i], Y], feed_dict={X_data: test_data[i], Y_data: test_label[i], keep_prob: 1.0})
                acc1 = models.get_accuracy(hypo, y_real, mode=0, one_hot=True)
                acc2 = models.get_accuracy(hypo, y_real, mode=1, one_hot=True)
                print(f"When {i} step{j} accuracy1 = {acc1} acc2 = {acc2}")
            sess.run(train_nn[i], feed_dict={X_data: train_data[i], Y_data: train_label[i], keep_prob: 0.8})


