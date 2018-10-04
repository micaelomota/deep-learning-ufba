def gradient_descent_step(b0, w0, batch, learning_rate):
    # compute gradients
    b_grad = 0
    w_grad = 0
    N = len(batch)
    for i in range(N):
        x = batch[i, 0]
        y = batch[i, 1]
        b_grad += (2.0/N)*(w0*x + b0 - y)
        w_grad += (2.0/N)*x*(w0*x + b0 - y)

    # update parameters
    b1 = b0 - (learning_rate * b_grad)
    w1 = w0 - (learning_rate * w_grad)

    return b1, w1
