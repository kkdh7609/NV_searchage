import tensorflow as tf
import numpy as np

# setting weights looks like size
def set_weight(size):
    return tf.Variable(tf.random_normal(size, stddev=0.1))


# setting bias looks like size
def set_bias(size):
    return tf.Variable(tf.random_normal(size, stddev=0.2))


# Linear regression model
class LinReg:
    def __init__(self):
        self.W = set_weight([4, 1])
        self.b = set_bias([1])

    def forward(self, x):
        h = tf.matmul(x, self.W) + self.b
        return h

    def coefs(self):
        return self.W, self.b


# Logistic regression model
class LogReg:
    def __init__(self, label_size=5):
        self.W = set_weight([4, label_size])
        self.b = set_bias([label_size])

    def forward(self, x):
        h = tf.matmul(x, self.W) + self.b
        h = tf.nn.softmax(h)
        return h


# Deep Neural Network
class NvNN:
    # one_hot means whether inputs are one hot form. out_one_hot means whether output should be one hot form.
    def __init__(self, x_size=4, one_hot=False, out_one_hot=False, label_size=5):
        self.l_size = [10, 20, 10, 1]
        self.out_one_hot = out_one_hot
        if one_hot and not out_one_hot:
            self.l_size = [70, 35, 10, 1]
            self.x_size = x_size * 20
        elif one_hot and out_one_hot:
            self.l_size = [80, 40, 20, label_size]
            self.x_size = x_size * 20
        elif not one_hot and out_one_hot:
            self.l_size = [10, 25, 15, label_size]
            self.x_size = x_size
        else:
            self.x_size = x_size
        self.W1 = set_weight([self.x_size, self.l_size[0]])
        self.b1 = set_bias([self.l_size[0]])

        self.W2 = set_weight([self.l_size[0], self.l_size[1]])
        self.b2 = set_bias([self.l_size[1]])

        self.W3 = set_weight([self.l_size[1], self.l_size[2]])
        self.b3 = set_bias([self.l_size[2]])

        self.W4 = set_weight([self.l_size[2], self.l_size[3]])
        self.b4 = set_bias([self.l_size[3]])

    def forward(self, x, keep_prob):
        L1 = tf.matmul(x, self.W1) + self.b1
        L1 = tf.nn.leaky_relu(L1)
        L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

        L2 = tf.matmul(L1, self.W2) + self.b2
        L2 = tf.nn.leaky_relu(L2)
        L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

        L3 = tf.matmul(L2, self.W3) + self.b3
        L3 = tf.nn.leaky_relu(L3)
        L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

        L4 = tf.matmul(L3, self.W4) + self.b4
        if self.out_one_hot:
            L4 = tf.nn.softmax(L4)
            return L4
        else:
            L4 = tf.nn.tanh(L4)
            L4 = (L4 + 1) * 5
            return L4


class NvRNN:
    def __init__(self):
        self.U = set_weight([21, 15])
        self.W = set_weight([15, 15])
        self.b1 = set_bias([15])

        self.V = set_weight([15, 10])
        self.b2 = set_bias([10])

    def forward(self, x1, x2, x3, x4, x5, keep_prob):
        L1 = tf.matmul(x1, self.U)
        L1 = L1 + self.b1
        L1 = tf.nn.leaky_relu(L1)
        L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

        L2 = tf.matmul(L1, self.W) + tf.matmul(x2, self.U)
        L2 = L2 + self.b1
        L2 = tf.nn.leaky_relu(L2)
        L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

        L3 = tf.matmul(L2, self.W) + tf.matmul(x3, self.U)
        L3 = L3 + self.b1
        L3 = tf.nn.leaky_relu(L3)
        L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

        L4 = tf.matmul(L3, self.W) + tf.matmul(x4, self.U)
        L4 = L4 + self.b1
        L4 = tf.nn.leaky_relu(L4)
        L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

        L5 = tf.matmul(L4, self.W) + tf.matmul(x5, self.U)
        L5 = L5 + self.b1
        output = tf.matmul(L5, self.V) + self.b2
        output = tf.nn.softmax(output)
        return output


# mode 0 - Accuracy for accurate value
# mode 1 - Accuracy for mitigated criteria
def get_accuracy(hypo, y_real, mode=0, one_hot=False, y_axis=1):
    t = hypo
    y_max = y_real
    if one_hot:
        t = np.argmax(t, axis=1)
        y_max = np.argmax(y_max, axis=1)

    if mode == 0:
        prediction = np.equal(t, y_max)
        acc = np.mean(prediction)
        return acc

    elif mode == 1:
        cp1 = np.equal(np.argsort(hypo, axis=1)[:, -1], np.argmax(y_real, axis=y_axis))
        cp2 = np.equal(np.argsort(hypo, axis=1)[:, -2], np.argmax(y_real, axis=y_axis))
        cp3 = np.equal(np.argsort(hypo, axis=1)[:, -3], np.argmax(y_real, axis=y_axis))
        cp4 = np.equal(np.argsort(hypo, axis=1)[:, -4], np.argmax(y_real, axis=y_axis))
        acc = np.mean(cp1 + cp2)
        return acc

    else:
        prediction1 = np.equal(t, y_max)
        prediction2 = np.equal(t, [i - int(i % 2) for i in y_max])
        prediction3 = np.equal([i - int(i % 2) for i in t], y_max)
        total_prediction = prediction1 + prediction2 + prediction3
        total_prediction = total_prediction - (total_prediction / 2).astype(int)
        total_prediction = total_prediction - (total_prediction / 2).astype(int)
        acc = np.mean(total_prediction)
        return acc