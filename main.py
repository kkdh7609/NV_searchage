import models
import tensorflow as tf
import numpy as np
import csv
import random

all_data = []     # for shuffling data
data_time = []
train_data = []
train_label = []

with open('train_set.csv', 'r') as f:
    rd = csv.reader(f)
    for line in rd:
        all_data.append(line)

# shuffle the data
random.shuffle(all_data)

for data in all_data:
    data_time.append([data[0]])
    train_data.append([float(i) for i in data[2: -1]])
    train_label.append([float(line[-1]) - 1])

# Make arrays to numpy
train_data = np.array(train_data)
train_label = np.array(train_label)


print("Data setting is done!")

