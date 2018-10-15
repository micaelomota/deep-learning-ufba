import tensorflow as tf
import numpy as np
from src import dataloader


data, labels, classes = dataloader.loadData("data_part1/train/")
td, tl, vd, vl = dataloader.splitValidation(data, labels, 10)

# normalizing data
td = np.reshape(td, (len(td), 77*71))/255
vd = np.reshape(vd, (len(vd), 77*71))/255

# 10 dimensoes para o y
tl10 = np.zeros((len(tl), 10))
for i in range(len(tl)):
    tl10[i][int(tl[i])] = 1

vl10 = np.zeros((len(vl), 10))
for i in range(len(vl)):
    vl10[i][int(vl[i])] = 1