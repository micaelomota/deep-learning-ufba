import numpy as np
from src import dataloader
import cv2

def reLU(z):
    return [x if x > 0 else 0 for x in z]

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def gradient_descent_step(b0, w0, x, y, learning_rate):
    b_grad = np.zeros(10)
    w_grad = np.zeros((len(w0), 10))
    loss = np.zeros(10)

    #print("w_grad shape: {}".format(w_grad.shape))

    N = len(x)
    for i in range(N): # x[i] -> y[i]
        y_ = sigmoid(np.dot(x[i], w0) + b0)
        #print("y_ shape: {}".format(y_.shape))
        loss += (y_ - y[i])**2

        # derivada de E em W
        dE = (y_ - y[i]) * (y_*(1-y_))
        #print("dE_ shape: {}".format(dE.shape))

        reshapeX = np.reshape(x[i], (len(x[i]), 1))
        reshapeDE = np.reshape(dE, (1, len(dE)))
        
        b_grad += dE
        w_grad += np.dot(reshapeX, reshapeDE)
        # for j in range(len(x[i])):
        #     w_grad[j] += x[i][j]*dE
        

    w_grad = w_grad/float(N)
    b_grad = b_grad/float(N)
    loss = loss/float(N)

    b1 = b0 - (learning_rate * b_grad)
    w1 = w0 - (learning_rate * w_grad)
    #print(b1)
    return b1, w1, loss

def validate(x, y, w0, b0):
    ok = 0
    for i in range(len(x)):
        y_ = sigmoid(np.dot(x[i], w0) + b0)
        shot = np.argmax(y_)
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

    epoch = 130
    learning_rate = 0.1
    batch_size = 20

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


    print("Trainning...")
    maxAc = 0
    for i in range(epoch):
        if (i+1)% 40 == 0:
            learning_rate = learning_rate/10

        loss = 0
        for j in range (len(td)//batch_size):
            l = j*batch_size
            r = min(l+batch_size, len(td))
            #print("batch {} from {} to {}".format(j, l, r))
            b, w, loss = gradient_descent_step(b, w, td[l:r], tl10[l:r], learning_rate)

        ac = validate(vd, vl10, w, b)
        if (ac > maxAc):
            #print("maxAc - Saving model...")
            maxAc = ac
            np.save("models/logisticRegression/w", w)
            np.save("models/logisticRegression/b", b)

        print("{}/{} - ac: {} - loss: {} - learningRate: {} - maxAc: {}".format(i+1, epoch, ac, np.mean(loss), learning_rate, maxAc))


    

	# for i in range(0, 4):
	# 	cv2.imshow('imagem' + str(i), td[i])
		
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	#print(td)
