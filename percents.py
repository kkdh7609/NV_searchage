import csv
import numpy as np

all_data = []     # for shuffling data
train_data = []
train_label = []

with open('train_set211.csv', 'r') as f:
    rd = csv.reader(f)
    for line in rd:
        train_data.append(line[2: -1])
        train_label.append(line[-1])

print(len(train_data))
data_value = []
label_value = []

mul_array = [10000, 100000, 1000000, 10000000]
train_data = np.array(train_data, dtype=float)
for i in train_data:
    mul_num = mul_array[0] * (i[0] // 10)
    if mul_num == 0:
        mul_num = 1
    mul_num = mul_num * mul_array[1] * (i[1] // 10)
    if mul_num == 0:
        mul_num = 1
    mul_num = mul_num * mul_array[2] * (i[2] // 10)
    if mul_num == 0:
        mul_num = 1
    mul_num = mul_num * mul_array[3] * (i[3] // 10)
    if mul_num == 0:
        mul_num = 1
    i[0] = i[0] - (i[0] // 10)
    i[1] = i[1] - (i[1] // 10)
    i[2] = i[2] - (i[2] // 10)
    i[3] = i[3] - (i[3] // 10)
    if mul_num is 0:
        mul_num = 1
    #print(mul_num)
    data_value.append(((int(i[0]) - 1) * 1000 + (int(i[1]) - 1) * 100 + (int(i[2]) - 1) * 10 + (int(i[3]) - 1)) * mul_num)

for j in train_label:
    label_value.append(int(j[0]))

data_value = np.array(data_value)
label_value = np.array(label_value)

sorted_dt = data_value[np.argsort(data_value)]
sorted_lb = label_value[np.argsort(data_value)]

labels = []
mx = 0
max_index = []
ct = 0

for i in range(len(sorted_dt)):
    if i == 0:
        labels.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        labels[-1][sorted_lb[i]-1] += 1
        ct = 1
    elif sorted_dt[i-1] == sorted_dt[i]:
        labels[-1][sorted_lb[i] - 1] += 1
        ct += 1
    else:
        if ct > 100:
            max_index.append(i)
        labels.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        labels[-1][sorted_lb[i] - 1] += 1
        ct = 1

print(mx)
print(max_index)

for i in max_index:
    print(sorted_dt[i])

trytry = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
labels = np.array(labels)
for i in labels:
    for j in range(9, -1, -1):
        if (i == 0).sum() == j:
            trytry[9-j] += 1

print(trytry)