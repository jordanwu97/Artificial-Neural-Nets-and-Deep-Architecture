import numpy as np
import matplotlib.pyplot as plt
import _3_1_1_lin_seperable_data
import _3_1_2_single_layer_perceptron

def sigmoid(x):
    return (2 / (1 + np.exp(-x))) - 1

def d_sigmoid(x):
    return (1+sigmoid(x)) * (1-sigmoid(x)) / 2

v_sigmoid = np.vectorize(lambda x: (2 / (1 + np.exp(-x))) - 1)
v_d_sigmoid = np.vectorize(lambda x: (1+sigmoid(x)) * (1-sigmoid(x)) / 2)

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

class Layer():
    def __init__(self, activation, d_activation):
        self.activation = activation
        self.d_activation = d_activation

    def forward(self,W,X,pad=False):
        """
        Returns layer_in, layer_out, layer_phid
        """
        self.IN = _3_1_2_single_layer_perceptron.forwardPass(W,X)
        self.OUT = self.activation(self.IN)
        self.PHID = self.d_activation(self.IN)

        if pad:
            padding = np.ones(self.OUT.shape[1])
            self.OUT = np.vstack([self.OUT, padding])
            self.PHID = np.vstack([self.PHID, padding])

        return self.OUT

    def d_out_hidden(self, d_in_next, w_next):
        return np.matmul(np.transpose(w_next), d_in_next)
    
    def d_in(self, d_out):
        return d_out * self.PHID

def runNN():

    eta = 0.001
    ndata = 100
    n_hidden = 4

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

    layer1 = Layer(activation=v_sigmoid, d_activation=v_d_sigmoid)
    layer2 = Layer(activation=v_sigmoid, d_activation=v_d_sigmoid)

    for epoch in range(100):

        
        
        #         +-------+                +-------+
        #  X(3x1)+>Affine1+--->Hj_in(4x1)+->Sigmoid+->Hj_out(4x1)
        #         +-------+                +-------+
        #                                  +---------+
        #          Wji(4x3)                |D_sigmoid+->Hj_phid(4x1)
        #                                  +---------+        

        h_out = layer1.forward(Wji, X, pad=True)

        #         +-------+                +-------+
        #  Hj_out+>Affine1+--->Yk_in(1x1)+->Sigmoid+->Yk_out(1x1)
        #  Biased +-------+                +-------+
        #  (5x1)                           +---------+
        #          Wji(1x5)                |D_sigmoid+->Yk_phid(1x1)
        #                                  +---------+

        y_out = layer2.forward(Wkj, h_out, pad=False)

        classification_percent = np.sum(np.sign(y_out) == T)/len(T)

        error = _3_1_2_single_layer_perceptron.error(T, y_out)
        losses.append(error)

        print(f"Epoch: {epoch} Error: {error} Percentage: {classification_percent}")

        eta = 0.001
        d_y_out = y_out - T
        d_y_in = layer2.d_in(d_y_out)
        DELTA_Wkj = DELTA(eta, d_y_in, h_out)

        d_h_out = layer1.d_out_hidden(d_y_in, Wkj)
        d_h_in = layer1.d_in(d_h_out)
        DELTA_Wji = DELTA(eta, d_h_in, X)[:-1,:]

        Wji = Wji + DELTA_Wji
        Wkj = Wkj + DELTA_Wkj

        print (max(np.max(DELTA_Wji), np.max(DELTA_Wji)))

        if max(np.max(DELTA_Wji), np.max(DELTA_Wji)) < 10^-6:
            break

    return losses

if __name__ == "__main__":
    import utility
    utility.plotLearningCurve(("train",runNN()))
