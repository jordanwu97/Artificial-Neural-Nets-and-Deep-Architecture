import numpy as np
import matplotlib.pyplot as plt
import _3_2_1_two_layer_network
import _3_1_2_single_layer_perceptron


def generate_Data(n):
    A = np.array([[-1 for x in range(n)] for y in range(n)])

    for element in range(n):
        A[element][element] = 1

    return A


def runNN():
    eta = 0.01
    ndata = 100

    vsigmoid = np.vectorize(_3_2_1_two_layer_network.sigmoid)
    vd_sigmoid = np.vectorize(_3_2_1_two_layer_network.d_sigmoid)


    #define nodes
    n_hidden = 3
    n_first = 8
    n_last = n_first

    # Training Data sets A
    A = generate_Data(n_first)
    print(A.shape[0])
    X = np.vstack([A, np.ones(A.shape[1])])

    #Targets = Training Data

    #random weights
    Wji = _3_1_2_single_layer_perceptron.initializeWeights(X.shape[0], n_hidden)
    Wkj = _3_1_2_single_layer_perceptron.initializeWeights(n_hidden + 1, n_last)

    losses = []

    for epoch in range(5000):

        Hj_in = _3_1_2_single_layer_perceptron.forwardPass(Wji, X)
        Hj_bias = np.ones(Hj_in.shape[1])
        Hj_out = np.vstack([vsigmoid(Hj_in), Hj_bias])
        Hj_phid = np.vstack([vd_sigmoid(Hj_in), Hj_bias])

        Yk_in = _3_1_2_single_layer_perceptron.forwardPass(Wkj, Hj_out)
        Yk_out = vsigmoid(Yk_in)

        Yk_phid = vd_sigmoid(Yk_in)

        classification_percent = np.sum(np.sign(Yk_out) == A)/len(A) ##T

        error = _3_1_2_single_layer_perceptron.error(A,Yk_out)
        losses.append(error)

        print(f"Epoch: {epoch} Error: {error}  Percentage: {classification_percent}")


        d_k = _3_2_1_two_layer_network.dFinal(A, Yk_out, Yk_phid)
        DELTA_Wkj = _3_2_1_two_layer_network.DELTA(eta, d_k, Hj_out)

        d_j = _3_2_1_two_layer_network.dHidden(d_k, Wkj, Hj_phid)
        DELTA_Wji = _3_2_1_two_layer_network.DELTA(eta, d_j, X)[:-1, :]

        Wji = Wji + DELTA_Wji
        Wkj = Wkj + DELTA_Wkj

        print(max(np.max(DELTA_Wji), np.max(DELTA_Wji)))

        if max(np.max(DELTA_Wji), np.max(DELTA_Wji)) < 10 ^ -6:
            break

        #corresponding internal code (representing binary conversion)
        #-> should be impossible for 2 hidden layer (can be seen at Yk_out)
        #with enough steps (50K is enough) error goes close to 0.
        #network "encodes" Data in strength of hj_out an in incredible high weights of Wkj.
        #hj_out is either 0.3 0.5 or 1 (and those negated)

        if epoch == 4999:
            print(Wji)
            print(Hj_out)
            print(Wkj)
            print(Yk_out)


    return losses

if __name__ == "__main__":
    import utility
    utility.plotLearningCurve(runNN())
