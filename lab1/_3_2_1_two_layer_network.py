import numpy as np
import matplotlib.pyplot as plt
import _3_1_1_lin_seperable_data
import _3_1_2_single_layer_perceptron
import math
import time

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

def runNN(n_hidden_max):

    losses = [] #training set
    losses_0 = [] #validation set

    ##(swap for more hidden nodes)
    for n_hidden in range(2,3): #(1,n_hidden_max+1):
        #print(n_hidden)
        eta = 0.001
        ndata = 50
        #n_hidden = 3

        # Training Data sets A and B
        mA, sigmaA = [0.0, -0.1], 0.3
        mB, sigmaB = [1.0, 0.3], 0.2
        mC, sigmaC = [-1.0, -0.3], 0.2
        #mD, sigmaD = [3, 4], 0.7

        ## Really quick just added a second set C to check this actually works on lin non-seperable.
        ## @Bryan please plug in the correct data set
        A = _3_1_1_lin_seperable_data.generateData(2, mA, sigmaA, math.floor(ndata*2))
        B = _3_1_1_lin_seperable_data.generateData(2, mB, sigmaB, ndata)
        C = _3_1_1_lin_seperable_data.generateData(2, mC, sigmaC, ndata)
       # D = _3_1_1_lin_seperable_data.generateData(2, mD, sigmaD, ndata)
        B = np.hstack([B,C])
       # B = np.hstack([B,D])

        #create validation set
        A_0 = A[:,0]
        B_0 = B[:,0]
        for _i in range(int(round(ndata *2 * .25))-1):
            _e = np.random.randint(A.shape[1])
            A_0 = np.vstack([A_0, A[:, _e]])
            A = np.delete(A, _e, axis=1)
            B_0 = np.vstack([B_0, B[:, _e]])
            B = np.delete(B, _e, axis=1)


        A_0 = np.transpose(A_0)
        B_0 = np.transpose(B_0)

        X = np.hstack([A, B])
        X_0 = np.hstack([A_0, B_0])

        # Add biasing term
        X = np.vstack([X, np.ones(X.shape[1])])
        X_0 = np.vstack([X_0, np.ones(X_0.shape[1])])

        # Targets
        T_A, T_B = np.ones(A.shape[1]), -1 * np.ones(B.shape[1])
        T = np.hstack([T_A, T_B])
        T_A_0, T_B_0 = np.ones(A_0.shape[1]), -1 * np.ones(B_0.shape[1])
        T_0 = np.hstack([T_A_0, T_B_0])


        # Wj = weights at layer j
        # Wk = weights at layer k
        # Wkj has a + 1 to reflect the weights associated with the bias
        Wji = _3_1_2_single_layer_perceptron.initializeWeights(X.shape[0], n_hidden)
        Wkj = _3_1_2_single_layer_perceptron.initializeWeights(n_hidden + 1, 1)



        layer1 = Layer(activation=v_sigmoid, d_activation=v_d_sigmoid)
        layer2 = Layer(activation=v_sigmoid, d_activation=v_d_sigmoid)

        # For matplotlib window
        maxX = X[:, np.argmax(X[0])][0] + 0.5
        minX = X[:, np.argmin(X[0])][0] - 0.5
        maxY = X[:, np.argmax(X[1])][1] + 0.5
        minY = X[:, np.argmin(X[1])][1] - 0.5

        eta = 0.01

        plt.ion()
        start = time.time()

        for epoch in range(1000):


            def batch_learning(Wji, Wkj):
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
                error = error/len(T)
                losses.append(error)

                print(f"Epoch: {epoch} Error: {error} Percentage: {classification_percent}")

                d_y_out = y_out - T
                d_y_in = layer2.d_in(d_y_out)
                DELTA_Wkj = DELTA(eta, d_y_in, h_out)

                d_h_out = layer1.d_out_hidden(d_y_in, Wkj)
                d_h_in = layer1.d_in(d_h_out)
                DELTA_Wji = DELTA(eta, d_h_in, X)[:-1,:]

                h_out_0 = layer1.forward(Wji, X_0, pad=True)
                y_out_0 = layer2.forward(Wkj, h_out_0, pad=False)
                error_0 = _3_1_2_single_layer_perceptron.error(T_0, y_out_0)
                error_0 = error_0 / len(T_0)
                losses_0.append(error_0)

                Wji = Wji + DELTA_Wji
                Wkj = Wkj + DELTA_Wkj
                return Wji, Wkj
            Wji, Wkj = batch_learning(Wji, Wkj)

            #print (max(np.max(DELTA_Wji), np.max(DELTA_Wji)))



            def sequential_learning(Wji, Wkj):

                Wji_old = np.copy(Wji)
                Wkj_old = np.copy(Wkj)

                classification_percent = 0.0
                error = 0.0

                for i in range(len(T)):

                    X_sample = np.transpose(np.atleast_2d(X[:, i]))
                    T_sample = np.atleast_2d(T[i])


                    #pass forward layer 1
                    layer_1_IN = _3_1_2_single_layer_perceptron.forwardPass(Wji, X_sample)
                    h_out = v_sigmoid(layer_1_IN)
                    layer_1_PHID = v_d_sigmoid(layer_1_IN)

                    #bias layer 1
                    padding = np.ones(h_out.shape[1])
                    h_out = np.vstack([h_out, padding])
                    layer_1_PHID = np.vstack([layer_1_PHID, padding])

                    #pass forward layer 2
                    layer_2_IN = _3_1_2_single_layer_perceptron.forwardPass(Wkj, h_out)
                    y_out = v_sigmoid(layer_2_IN)
                    layer_2_PHID = v_d_sigmoid(layer_2_IN)


                    classification_percent += np.sum(np.sign(y_out) == T_sample)

                    error += _3_1_2_single_layer_perceptron.error(T_sample, y_out)

                    #backprop

                    d_y_out = y_out - T_sample
                    d_y_in = d_y_out * layer_2_PHID
                    DELTA_Wkj = DELTA(eta, d_y_in, h_out)

                    d_h_out = np.matmul(np.transpose(Wkj), d_y_in)
                    d_h_in = d_h_out * layer_1_PHID
                    DELTA_Wji = DELTA(eta, d_h_in, X_sample)[:-1, :]

                    Wji = Wji + DELTA_Wji
                    Wkj = Wkj + DELTA_Wkj


                print(f"Epoch: {epoch} Error: {error} Percentage: {classification_percent/len(T)}")
                error = error / len(T)
                losses.append(error)

                h_out_0 = layer1.forward(Wji_old, X_0, pad=True)
                y_out_0 = layer2.forward(Wkj_old, h_out_0, pad=False)
                error_0 = _3_1_2_single_layer_perceptron.error(T_0, y_out_0)
                error_0 = error_0 / len(T_0)
                losses_0.append(error_0)

                return Wji, Wkj
            #Wji, Wkj = sequential_learning(Wji, Wkj)







            __x = np.arange(-5, 5, 0.5)

            def decisionBoundary(x, normal):
                # 0 = w0x + w1y + w2
                # y = (-w0x - w2) / w1
                if normal.shape[0] == 3:
                    return (-normal[0] * x - normal[2]) / normal[1]
                else:
                    return (-normal[0] * x) / normal[1]

            def plot(n_hidden_layer):
                plt.clf()
                plt.ylim(minY, maxY)
                plt.xlim(minX, maxX)

                plt.plot(A[0], A[1], "ro", B[0], B[1], "bo")
                for _i in range(n_hidden_layer):
                    plt.plot(__x, decisionBoundary(__x, np.transpose(Wji[_i,:])), label=f"hidden node: {_i}")
                plt.title("Grid")
                plt.xlabel("X-Coord")
                plt.ylabel("Y-Coord")
                plt.legend()
                plt.show()
                plt.pause(0.0000001)
            #plot()

        elapsed = time.time() - start

        print(f"\nElapsed: {elapsed}")

        print(Wji)
        print(Wkj)

        plt.ioff()
        if (n_hidden == 2):
            plot(n_hidden)

    return losses, losses_0

if __name__ == "__main__":
    import utility
    max_test = 5

    losses, losses_0 = runNN(max_test)


    utility.plotLearningCurves((["training set", "validation set"], losses, losses_0))

#part 1 different amounts of hidden nodes

#    list_losses = np.random.randn(max_test, 1000)
#    for _i in range(max_test):
#        list_losses[_i, :] = losses[0 + 1000 * _i:1000 + 1000 * _i]

    #utility.plotLearningCurves(
    #    (["error_1", "error_2", "error_3", "error_4", "error_5"],
    #     list_losses[0,:], list_losses[1,:], list_losses[2,:], list_losses[3,:], list_losses[4,:]))
