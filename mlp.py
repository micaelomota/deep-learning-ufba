import numpy as np
from src import dataloader
import cv2

epoch = 90
learning_rate = 0.1
batch_size = 10
hidden_layer_neurons = 64


def reLU(z):
    return [x if x > 0 else 0 for x in z]

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def gradient_descent_step(bj, wj, bk, wk, x, y, learning_rate):
    N = len(x)
    wk_grad = np.zeros((10, hidden_layer_neurons))
    bk_grad = np.zeros(10)

    wj_grad = np.zeros((hidden_layer_neurons, len(x[0])))
    bj_grad = np.zeros(hidden_layer_neurons)

    loss = np.zeros(10)

    for i in range(N):
        """ layer J """
        j_activation = sigmoid(np.dot(x[i], wj) + bj)

        k_activation = sigmoid(np.dot(j_activation, wk) + bk)

        loss += (k_activation - y[i])**2

        # derivada de E em W
        dk = (k_activation - y[i]) * (k_activation * (1 - k_activation))
        """ layer k gradient """
        wk_grad += np.dot(np.reshape(dk, (10, 1)), np.reshape(j_activation, (1, hidden_layer_neurons)))
        bk_grad += dk

        dj = np.dot(dk, wk.transpose())

        wj_grad += np.dot(np.reshape(dj, (hidden_layer_neurons, 1)), np.reshape(x[i], (1, len(x[i]))))
        bj_grad += dj

        #print("wj shape: {}".format(wj_grad.shape))

    b1 = bk - (learning_rate * bk_grad/float(N))
    w1 = wk - (learning_rate * wk_grad.transpose()/float(N))

    b2 = bj - (learning_rate * bj_grad/float(N))    
    w2 = wj - (learning_rate * wj_grad.transpose()/float(N))
    loss = loss/float(N)

    return b2, w2, b1, w1, loss

def inference(x, wj, bj, wk, bk):
    j_activation = sigmoid(np.dot(x, wj) + bj)
    y_ = sigmoid(np.dot(j_activation, wk) + bk)

    return np.argmax(y_)

def validate(x, y, wj, bj, wk, bk):
    ok = 0
    for i in range(len(x)):
        shot = inference(x[i], wj, bj, wk, bk)
        if (y[i][shot] == 1):
            ok += 1

    return ok/len(x)


if __name__ == '__main__': # main here
    data, labels, classes = dataloader.loadData("data_part1/train/")
    td, tl, vd, vl = dataloader.splitValidation(data, labels, 10)

    # normalizing train and validation data [0, 1]
    td = np.reshape(td, (len(td), 77*71))/255
    vd = np.reshape(vd, (len(vd), 77*71))/255
    
    # 10 dimensoes para os pesos
    wj = np.random.uniform(-0.1, 0.1, (len(td[0]), hidden_layer_neurons))
    bj = np.zeros(hidden_layer_neurons)
    
    wk = np.random.uniform(-0.1, 0.1, (hidden_layer_neurons, 10))
    bk = np.zeros(10)

    # 10 dimensoes para o y
    tl10 = np.zeros((len(tl), 10))
    for i in range(len(tl)):
        tl10[i][int(tl[i])] = 1


    vl10 = np.zeros((len(vl), 10))
    for i in range(len(vl)):
        vl10[i][int(vl[i])] = 1


    print("Network: wj: {}; bj: {}; wk: {}; bk: {}".format(wj.shape, bj.shape, wk.shape, bk.shape))

    print("Trainning...")
    maxAc = 0
    for i in range(epoch):
        if (i+1)%40 == 0:
            learning_rate = learning_rate/10
            
        for j in range (len(td)//batch_size):
            l = j*batch_size
            r = min(l+batch_size, len(td))
            #print("batch {} from {} to {}".format(j, l, r))
            bj, wj, bk, wk, loss = gradient_descent_step(bj, wj, bk, wk, td[l:r], tl10[l:r], learning_rate)

        ac = validate(vd, vl10, wj, bj, wk, bk)
        
        if (ac > maxAc):
            maxAc = ac
            np.save("models/mlp/wj", wj)
            np.save("models/mlp/bj", bj)
            np.save("models/mlp/wk", wk)
            np.save("models/mlp/bk", bk)
        
        print("epoch: {}/{} - ac: {} - loss: {} - learningRate: {} - maxAc: {}".format(i+1, epoch, ac, np.mean(loss), learning_rate, maxAc))


	# for i in range(0, 4):
	# 	cv2.imshow('imagem' + str(i), td[i])
		
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	#print(td)
