import numpy as np
import matplotlib.pyplot as plt
import _3_1_1_lin_seperable_data
import _3_1_2_single_layer_perceptron

def sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

def d_sigmoid(x):
    return (1+sigmoid(x)) * (1-sigmoid(x)) / 2

def dFinal(t, o, phi_final):
    """
    Returns the delta at the final layer
    """
    return (o - t) * phi_final

def dHidden(d_next, w_layer, phi_hidden):
    """
    Returns the delta at a layer
    """
    return np.matmul(np.transpose(w_layer), d_next) * phi_hidden

def DELTA(eta, d, X):
    """
    Returns the delta(triangle) for weights
    """
    return -eta * np.matmul(d, np.transpose(X))

def runNN():

    eta = 0.001
    ndata = 100
    n_hidden = 4

    vsigmoid = np.vectorize(sigmoid)
    vd_sigmoid = np.vectorize(d_sigmoid)

    # Training Data sets A and B
    mA, sigmaA = [1.0, 0.5], 0.5
    mB, sigmaB = [-2.0, 0.0], 0.5
    mC, sigmaC = [5, 0.0], 0.5

    ## Really quick just added a second set C to check this actually works on lin non-seperable.
    ## @Bryan please plug in the correct data set
    A = _3_1_1_lin_seperable_data.generateData(2, mA, sigmaA, ndata)
    B = _3_1_1_lin_seperable_data.generateData(2, mB, sigmaB, ndata)
    C = _3_1_1_lin_seperable_data.generateData(2, mC, sigmaC, ndata)
    B = np.hstack([B,C])
    X = np.hstack([A, B])
    # Add biasing term
    X = np.vstack([X, np.ones(X.shape[1])])

    # Targets
    T_A, T_B = np.ones(A.shape[1]), -1 * np.ones(B.shape[1])
    T = np.hstack([T_A, T_B])

    # Wj = weights at layer j
    # Wk = weights at layer k
    # Wkj has a + 1 to reflect the weights associated with the bias
    Wji = _3_1_2_single_layer_perceptron.initializeWeights(X.shape[0], n_hidden)
    Wkj = _3_1_2_single_layer_perceptron.initializeWeights(n_hidden + 1, 1)

    losses = []

    plt.plot(A[0], A[1], "ro", B[0], B[1], "bo")
    plt.show()

    for epoch in range(100):

        
        
        #         +-------+                +-------+
        #  X(3x1)+>Affine1+--->Hj_in(4x1)+->Sigmoid+->Hj_out(4x1)
        #         +-------+                +-------+
        #                                  +---------+
        #          Wji(4x3)                |D_sigmoid+->Hj_phid(4x1)
        #                                  +---------+        

        # forward pass
        # Hj_in = input to layer J
        # Hj_out = output at layer j
        # Yk= output at layer k
        Hj_in = _3_1_2_single_layer_perceptron.forwardPass(Wji,X)
        Hj_bias = np.ones(Hj_in.shape[1])
        Hj_out = np.vstack([vsigmoid(Hj_in),Hj_bias])
        Hj_phid = np.vstack([vd_sigmoid(Hj_in),Hj_bias])

        #         +-------+                +-------+
        #  Hj_out+>Affine1+--->Yk_in(1x1)+->Sigmoid+->Yk_out(1x1)
        #  Biased +-------+                +-------+
        #  (5x1)                           +---------+
        #          Wji(1x5)                |D_sigmoid+->Yk_phid(1x1)
        #                                  +---------+

        Yk_in = _3_1_2_single_layer_perceptron.forwardPass(Wkj,Hj_out)
        Yk_out= vsigmoid(Yk_in)
        Yk_phid = vd_sigmoid(Yk_in)

        classification_percent = np.sum(np.sign(Yk_out) == T)/len(T)

        error = _3_1_2_single_layer_perceptron.error(T, Yk_out)
        losses.append(error)

        print(f"Epoch: {epoch} Error: {error} Percentage: {classification_percent}")

        eta = 0.001
        d_k = dFinal(T, Yk_out, Yk_phid)
        DELTA_Wkj = DELTA(eta, d_k, Hj_out)

        d_j = dHidden(d_k, Wkj, Hj_phid)
        DELTA_Wji = DELTA(eta, d_j, X)[:-1,:]

        Wji = Wji + DELTA_Wji
        Wkj = Wkj + DELTA_Wkj

        print (max(np.max(DELTA_Wji), np.max(DELTA_Wji)))

        if max(np.max(DELTA_Wji), np.max(DELTA_Wji)) < 10^-6:
            break

    return losses

if __name__ == "__main__":
    import utility
    utility.plotLearningCurve(runNN())
