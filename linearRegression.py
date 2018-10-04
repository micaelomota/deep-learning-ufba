import numpy as np
from src import dataloader
import cv2

def gradient_descent_step(b0, w0, x, y, learning_rate):
    b_grad = 0
    w_grad = np.zeros(len(w0))
    N = len(x)
    loss = 0.
    for i in range(N): # x[i] -> y[i]
        y_ = np.dot(x[i], w0) + b0;
        deltinha = y_ - y[i]
        loss += 1./N * (y_ - y[i])**2
        w_grad += deltinha*x[i] * 2./N
        b_grad += 2./N * deltinha

    b1 = b0 - (learning_rate * b_grad)
    w1 = w0 - (learning_rate * w_grad)
    #print(b1)
    return b1, w1, loss

def validate(x, y, w0, b0):
    ok = 0
    for i in range(len(x)):
        y_ = np.dot(x[i], w0) + b0
        if (int(round(y_)) == y[i]):
            ok += 1
        
    return ok/len(x)


if __name__ == '__main__': # main here
    data, labels, classes = dataloader.loadData("data_part1/train/")
    td, tl, vd, vl = dataloader.splitValidation(data, labels, 10)

    # normalizing train and validation data [0, 1]
    td = np.reshape(td, (len(td), 77*71))
    vd = np.reshape(vd, (len(vd), 77*71))
    td = td/255.
    vd = vd/255.

    w = np.random.uniform(-1, 1, len(td[0]))
    b = 0

    epoch = 200
    learning_rate = 0.001
    batch_size = 100

    print("Trainning...")
    for i in range(1, epoch):
        
        v = 0
        for j in range (len(td)//batch_size):
            l = j*batch_size
            r = min(l+batch_size, len(td))
            b, w, loss = gradient_descent_step(b, w, td[l:r], tl[l:r], learning_rate)
            v = loss
            
            #print(loss)

        ac = validate(vd, vl, w, b)

        print("ac: {}; loss: {}".format(ac, v))


	# for i in range(0, 4):
	# 	cv2.imshow('imagem' + str(i), td[i])
		
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	#print(td)
