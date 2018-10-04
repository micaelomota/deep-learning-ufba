import numpy as np
from src import dataloader
import cv2

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def reLU(z):
    return [x if x > 0 else 0 for x in z]

def gradient_descent_step(b0, w0, x, y, learning_rate):
    b_grad = np.zeros(10)
    w_grad = np.zeros((len(w0), 10))
    
    N = len(x)

    for i in range(N): # x[i] -> y[i]
        y_ = reLU(np.dot(x[i], w0) + b0)

        loss = y_ - y[i]
        
        a = np.reshape(loss, (1, 10))
        b = np.reshape(x[i], (x[i].shape[0], 1))

        w_grad += a*b * 2./N
        b_grad += 2./N * loss


    b1 = b0 - (learning_rate * b_grad)
    w1 = w0 - (learning_rate * w_grad)
    #print(b1)
    return b1, w1

def validate(x, y, w0, b0):
    ok = 0
    for i in range(len(x)):
        y_ = np.dot(x[i], w0) + b0
        # print(y_)
        # print(y[i])
        shot = np.argmax(y_)
        # print(shot)
        if (y[i][shot] == 1):
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

    epoch = 200
    learning_rate = 0.01
    batch_size = 100

    # 10 dimensoes para os pesos
    w = np.random.uniform(-0.1, 0.1, (len(td[0]), 10))
    b = np.zeros(10)
    #print(w[0])
    
    # 10 dimensoes para o y
    tl10 = np.zeros((len(tl), 10))
    for i in range(len(tl)):
        tl10[i][int(tl[i])] = 1


    vl10 = np.zeros((len(vl), 10))
    for i in range(len(vl)):
        vl10[i][int(vl[i])] = 1

    #print(tl10)
    

    print("Trainning...")
    for i in range(1, epoch):
        for j in range (len(td)//batch_size):
            l = j*batch_size
            r = min(l+batch_size, len(td))
            b, w = gradient_descent_step(b, w, td[l:r], tl10[l:r], learning_rate)

        ac = validate(vd, vl10, w, b)

        print("ac: {};".format(ac))


	# for i in range(0, 4):
	# 	cv2.imshow('imagem' + str(i), td[i])
		
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	#print(td)
