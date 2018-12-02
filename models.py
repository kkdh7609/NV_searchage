import tensorflow as tf


# setting weights looks like size
def set_weight(size):
    return tf.Variable(tf.random_normal(size, stddev=0.1))


# setting bias looks like size
def set_bias(size):
    return tf.Variable(tf.random_normal(size, stddev=0.2))


# Linear regression model
class LinReg:
    def __init__(self):
        self.W = set_weight([5, 1])
        self.b = set_bias([1])

    def forward(self, x):
        h = tf.matmul(x, self.W) + self.b
        return h


# Logistic regression model
class LogReg:
    def __init__(self, x_size=5):
        self.W = set_weight([5, x_size])
        self.b = set_bias([x_size])

    def forward(self, x):
        h = tf.matmul(x, self.W) + self.b
        h = tf.nn.softmax(h)
        return h


# Deep Neural Network
class NvNN:
    # one_hot means whether inputs are one hot form. out_one_hot means whether output should be one hot form.
    def __init__(self, x_size=5, one_hot=False, out_one_hot=False):
        self.l_size = [10, 20, 10, 1]
        self.out_one_hot = out_one_hot
        if one_hot and not out_one_hot:
            self.l_size = [70, 35, 10, 1]
            self.x_size = x_size * 21
        elif one_hot and out_one_hot:
            self.l_size = [80, 40, 20, 10]
            self.x_size = x_size * 21
        elif not one_hot and out_one_hot:
            self.l_size = [10, 25, 15, 10]
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

