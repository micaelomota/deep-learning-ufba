import numpy as np
from src import dataloader
import cv2

def runLogisticRegression():
    w = np.load("models/logisticRegression/w.npy")
    b = np.load("models/logisticRegression/b.npy")

    data, names = dataloader.loadTestData('data_part1/test/')
    data = np.reshape(data, (len(data), 77*71))/255
    for i in range(len(data)):
        y_ = np.dot(data[i], w) + b
        shot = np.argmax(y_)

        print("{} {}".format(names[i], shot))

def runMlp():
    import mlp

    wj = np.load("models/mlp/wj.npy")
    bj = np.load("models/mlp/bj.npy")
    wk = np.load("models/mlp/wk.npy")
    bk = np.load("models/mlp/bk.npy")

    data, names = dataloader.loadTestData('data_part1/test/')
    data = np.reshape(data, (len(data), 77*71))/255

    for i in range(len(data)):
        print("{} {}".format(names[i], mlp.inference(data[i], wj, bj, wk, bk)))


#runMlp()
runLogisticRegression()