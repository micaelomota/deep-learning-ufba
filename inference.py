import numpy as np
from src import dataloader
import cv2

w = np.load("models/logisticRegression/w.npy")
b = np.load("models/logisticRegression/b.npy")

#imagePath = ""

#img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

def runTests():
    
    data, names = dataloader.loadTestData('data_part1/test/')
    data = np.reshape(data, (len(data), 77*71))
    for i in range(len(data)):
        y_ = np.dot(data[i], w) + b
        shot = np.argmax(y_)

        print("{} {}".format(names[i], shot))
