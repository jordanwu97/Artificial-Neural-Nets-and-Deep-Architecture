import numpy as np
import matplotlib.pyplot as plt
import part1 as p1
import nonlinear_exploration_3_1_3 as p313
import math

def sigmoid(x):
    return (1 - math.e**(-x))/(1 + math.e **(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

if __name__ == "__main__":

    eta = 0.001
    ndata = 100
    case = 3
    n_hidden = 3

    vsigmoid = np.vectorize(sigmoid)
    vd_sigmoid = np.vectorize(d_sigmoid)

    # Training Data sets A and B
    mA, sigmaA = [1.0, 0.3], 0.2
    mB, sigmaB = [0.0, -0.1], 0.3
    
    A = p313.generateNonlinearData(2, mA, sigmaA, ndata)
    B = p1.generateData(2, mB, sigmaB, ndata)

    # Subselect
    A_train, B_train, A_test, B_test  = p313.subselect( A, B, case)

    X = np.hstack([A_train, B_train])

    # Add biasing term
    X = np.vstack([X, np.ones(X.shape[1])])

    # Targets
    T_A, T_B = np.ones(A_train.shape[1]), -1 * np.ones(B_train.shape[1])
    T = np.hstack([T_A, T_B])

    # Wj = weights at layer j
    # Wk = weights at layer k
    Wji = p1.initializeWeights(X.shape[0], n_hidden)
    Wkj = p1.initializeWeights(n_hidden, 1)

#    for _ in range(ndata):
    for _ in range(1):

        # forward pass
        # Hj_in = input to layer J
        # Hj_out = output at layer j
        # Yk= output at layer k
        Hj_in = p1.forwardPass(Wji,X)
        Hj_out = vsigmoid(Hj_in)
        Yk_in = p1.forwardPass(Wkj,Hj_out)
        Yk_out= vsigmoid(Yk_in)

        # dk = delta_k in notes
        # delta_j in notes
        dk = np.matmul((Yk_out- T),  np.transpose(vd_sigmoid(Yk_in)))
        dj = np.sum(dk * np.matmul(Wkj,vd_sigmoid(Hj_in)))
        print dk, dj

        
        #print("p1.error:", p1.error(T, Y))
        #dW = -eta * np.matmul((Yout - T), np.transpose(X))
        #Wji = Wji + dW


    plt.plot(A_train[0], A_train[1], "ro", B_train[0], B_train[1], "bo")

    def line(x, normal):
        # 0 = w0x + w1y + w2
        # y = (-w0x - w2) / w1
        return (-normal[:,0] * x - normal[:,2]) / normal[:,1]

    maxX = X[:, np.argmax(X[1])][0]
    minX = X[:, np.argmin(X[1])][0]

    x = np.arange(minX, maxX, 0.05)

    #plt.plot(x, line(x, W1))

    #plt.show()
