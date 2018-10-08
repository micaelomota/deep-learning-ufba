import numpy as np
from src import dataloader


data, names = dataloader.loadTestData('data_part1/test/')
data = np.reshape(data, (len(data), 77*71))

w = np.load("w.npy")
b = np.load("b.npy")

for i in range(len(data)):
    y_ = np.dot(data[i], w) + b
    shot = np.argmax(y_)

    print("{} {}".format(names[i], shot))