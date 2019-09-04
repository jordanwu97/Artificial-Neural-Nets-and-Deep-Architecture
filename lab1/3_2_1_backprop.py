import numpy as np
import multilayer_3_2_1 as ml321

def dFinal(t, o, phi_final):
    return (o - t) * phi_final

def dHidden(d_next, w_layer, phi_hidden):
    return np.matmul(np.transpose(w_layer), d_next) * phi_hidden

def DELTA(eta, d, X):
    return -eta * np.matmul(d, np.transpose(X))

if __name__ == "__main__":
    
    eta = 0.001
    d2 = dFinal(T, Y2, phi2)
    DELTA2 = DELTA(eta, d2, Y1)

    d1 = dHidden(d2, W2, phi1)
    DELTA1 = Delta(eta, d1, X)