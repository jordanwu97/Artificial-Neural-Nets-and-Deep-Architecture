import numpy as np

import _3_2_1_two_layer_network
import _3_1_2_single_layer_perceptron

def gauss(x,y):
    return np.exp(-(x**2 + y**2)/10) - 0.5

if __name__ == "__main__":
    _x = np.arange(-5,5,0.5)
    _y = np.arange(-5,5,0.5)

    hidden_layer_nodes = 10

    T = _z = np.atleast_2d(gauss(_x,_y))

    patterns = np.vstack([_x,_y,np.ones(len(_x))])

    for epoch in range(1000):

        W_1 = _3_1_2_single_layer_perceptron.initializeWeights(3,hidden_layer_nodes)
        W_2 = _3_1_2_single_layer_perceptron.initializeWeights(hidden_layer_nodes + 1, 1)

        ## Layer 1
        In_1 = _3_1_2_single_layer_perceptron.forwardPass(W_1, patterns)
        Out_1 = np.vstack([_3_2_1_two_layer_network.sigmoid(In_1), np.ones(In_1.shape[1])])
        Dphi_1 = np.vstack([_3_2_1_two_layer_network.d_sigmoid(In_1), np.ones(In_1.shape[1])])

        ## Layer 2
        In_2 = _3_1_2_single_layer_perceptron.forwardPass(W_2, Out_1)
        Out_2 = In_2
        Dphi_2 = np.ones(In_2.shape)

        error = _3_1_2_single_layer_perceptron.error(T, Out_2)

        print (f"Epoch: {epoch} Error: {error}")

        eta = 0.000001

        d_2 = _3_2_1_two_layer_network.dFinal(T, Out_2, Dphi_2)
        DELTA_W2 = _3_2_1_two_layer_network.DELTA(eta, d_2, Out_1)

        d_1 = _3_2_1_two_layer_network.dHidden(d_2, W_2, Dphi_1)
        DELTA_W1 = _3_2_1_two_layer_network.DELTA(eta, d_1, patterns)[:-1,:]

        W_1 = W_1 + DELTA_W1
        W_2 = W_2 + DELTA_W2