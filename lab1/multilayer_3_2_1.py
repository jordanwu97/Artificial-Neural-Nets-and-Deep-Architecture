import numpy as np
import matplotlib.pyplot as plt
import part1 as p1
import nonlinear_exploration_3_1_3 as p313
import math

def sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1
    # return (1 - math.e**(-x))/(1 + math.e **(-x))

def d_sigmoid(x):
    return (1+sigmoid(x)) * (1-sigmoid(x)) / 2

def dFinal(t, o, phi_final):
    return (o - t) * phi_final

def dHidden(d_next, w_layer, phi_hidden):
    return np.matmul(np.transpose(w_layer), d_next) * phi_hidden

def DELTA(eta, d, X):
    return -eta * np.matmul(d, np.transpose(X))

def runNN():

    eta = 0.001
    ndata = 100
    case = 3
    n_hidden = 4

    vsigmoid = np.vectorize(sigmoid)
    vd_sigmoid = np.vectorize(d_sigmoid)

    # Training Data sets A and B
    mA, sigmaA = [1.0, 0.5], 0.5
    mB, sigmaB = [-2.0, 0.0], 0.5

    A = p1.generateData(2, mA, sigmaA, 100)
    B = p1.generateData(2, mB, sigmaB, 100)
    X = np.hstack([A, B])
    # Add biasing term
    X = np.vstack([X, np.ones(X.shape[1])])

    # Targets
    T_A, T_B = np.ones(A.shape[1]), -1 * np.ones(B.shape[1])
    T = np.hstack([T_A, T_B])

    # Wj = weights at layer j
    # Wk = weights at layer k
    # Wkj has a + 1 to reflect the weights associated with the bias
    Wji = p1.initializeWeights(X.shape[0], n_hidden)
    Wkj = p1.initializeWeights(n_hidden + 1, 1)

    losses = []

#    for _ in range(ndata):
    for epoch in range(100):

        # forward pass
        # Hj_in = input to layer J
        # Hj_out = output at layer j
        # Yk= output at layer k
        Hj_in = p1.forwardPass(Wji,X)
        Hj_bias = np.ones(Hj_in.shape[1])
        Hj_out = np.vstack([vsigmoid(Hj_in),Hj_bias])
        Hj_phid = np.vstack([vd_sigmoid(Hj_in),Hj_bias])

        Yk_in = p1.forwardPass(Wkj,Hj_out)
        Yk_out= vsigmoid(Yk_in)
        Yk_phid = vd_sigmoid(Yk_in)

        error = p1.error(T, Yk_out)
        losses.append(error)

        print(f"Epoch: {epoch}\nError: {error}")

        eta = 0.001
        d_k = dFinal(T, Yk_out, Yk_phid)
        DELTA_Wkj = DELTA(eta, d_k, Hj_out)

        d_j = dHidden(d_k, Wkj, Hj_phid)
        DELTA_Wji = DELTA(eta, d_j, X)[:-1,:]

        Wji = Wji + DELTA_Wji
        Wkj = Wkj + DELTA_Wkj

    print (Yk_out)

    return losses

if __name__ == "__main__":
    import utility
    utility.plotLearningCurve(runNN())
