
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
a = []
train_x = []
train_y = []

with open("./train_set_21.csv", "r") as f:
    reader = csv.reader(f)
    for line in reader:
        a.append(line)

a = np.array(a, dtype=np.int64)

train_x = a[:, 2:7]
train_y = a[:, 7:8]

m = np.mean(train_x)
m_y = np.mean(train_y)
o = np.std(train_x)
o_y = np.std(train_y)
z = np.zeros((95040, 5))
y = np.zeros((95040, 1))
print(len(train_x))
print(len(train_x[0]))

for i in range(95040):
    for j in range(5):
        z[i][j] = (train_x[i][j] - m) / o
    y[i] = (train_y[i] - m_y) / o_y
z = np.hstack([z, y])

with open("./normal10000.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for i in range(500):
        writer.writerow((z[i]))
