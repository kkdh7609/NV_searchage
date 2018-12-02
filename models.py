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
        self.W = set_weight([5, 1])
        self.b = set_bias([1])

    def forward(self, x):
        h = tf.matmul(x, self.W) + self.b
        return h
