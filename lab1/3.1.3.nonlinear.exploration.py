import numpy as np
import matplotlib.pyplot as plt
import _3_1_1_lin_seperable_data
import _3_1_2_single_layer_perceptron
import utility
import time


def generateNonlinearData(features, mean, sigma, n):
    """
    Returns 2 sets of data A and B
    Each dataset has shape (#features, #samples)
    """

    distA = sigma * np.random.randn(features, n)

    distA[0][:n//2] = distA[0][:n//2] + mean[0]
    distA[0][n//2:] = distA[0][n//2:] - mean[0]
    distA[1] = distA[1] + mean[1]

    return distA

def main():

    eta = 0.001
    ndata = 100
    subsampleCase = 3
    part = 2

    plt.ion()

    # 3.1.3 first part (Overlapping clouds)

    if part == 1:
        mA, sigmaA = [1.0, 0.3], 0.2
        mB, sigmaB = [0.5, 0.0], 0.3

        A = _3_1_1_lin_seperable_data.generateData(2, mA, sigmaA, ndata)
        B = _3_1_1_lin_seperable_data.generateData(2, mB, sigmaB, ndata)

        plt.plot(A[0],A[1],"bo")
        plt.plot(B[0],B[1],"ro")
        plt.show()


    # 3.1.3 second part (split class A into 2 parts)
    elif part == 2:
        #Training Data sets A and B
        mA, sigmaA = [1.0, 0.3], 0.2
        mB, sigmaB = [0.0, -0.1], 0.3

        A = generateNonlinearData(2, mA, sigmaA, ndata)
        B = _3_1_1_lin_seperable_data.generateData(2, mB, sigmaB, ndata)

        #plt.plot(A[0],A[1],"go")
        #plt.plot(B[0],B[1],"mo")
        #plt.show()


    # Subselection
    #subsampleCase == 0:
        A_0 = np.asarray([np.random.choice(i, int(round(ndata * .75)), replace=False) for i in A])
        B_0 = np.asarray([np.random.choice(i, int(round(ndata * .75)), replace=False) for i in B])
    #subsampleCase == 1:
        A_1 = np.asarray([np.random.choice(i, int(round(ndata * .50)), replace=False) for i in A])
        B_1 = B
    #subsampleCase == 2:
        A_2 = A
        B_2 = np.asarray([np.random.choice(i, int(round(ndata * .50)), replace=False) for i in B])
    #subsampleCase == 3:

        #TODO: apply to entire Class A not A[0]


        #A_g = greater then 0 in A[0,:]
        A_g = A[:,:50]
        A_l = A[:,50:]


        A_l_1 = A_l[:, np.random.choice(A_l.shape[1], int(round(A_l.shape[1] * 0.80)), replace=False)] #only take 80%
        A_g_1 = A_g[:, np.random.choice(A_g.shape[1], int(round(A_g.shape[1] * 0.20)), replace=False)] #only take 20%


        A_3 = np.hstack([A_l_1,A_g_1])
        B_3 = B



        X_0 = np.hstack([A_0, B_0])
        T_A_0, T_B_0 = np.ones(A_0.shape[1]), -1 * np.ones(B_0.shape[1])

        X_1 = np.hstack([A_1, B_1])
        T_A_1, T_B_1 = np.ones(A_1.shape[1]), -1 * np.ones(B_1.shape[1])

        X_2 = np.hstack([A_2, B_2])
        T_A_2, T_B_2 = np.ones(A_2.shape[1]), -1 * np.ones(B_2.shape[1])

        X_3 = np.hstack([A_3, B_3])
        T_A_3, T_B_3 = np.ones(A_3.shape[1]), -1 * np.ones(B_3.shape[1])




        # Add biasing term
        X_0 = np.vstack([X_0, np.ones(X_0.shape[1])])
        X_1 = np.vstack([X_1, np.ones(X_1.shape[1])])
        X_2 = np.vstack([X_2, np.ones(X_2.shape[1])])
        X_3 = np.vstack([X_3, np.ones(X_3.shape[1])])

        # Targets
        T_0 = np.hstack([T_A_0, T_B_0])
        T_1 = np.hstack([T_A_1, T_B_1])
        T_2 = np.hstack([T_A_2, T_B_2])
        T_3 = np.hstack([T_A_3, T_B_3])


        W_0 = _3_1_2_single_layer_perceptron.initializeWeights(X_0.shape[0], 1)
        W_1 = _3_1_2_single_layer_perceptron.initializeWeights(X_1.shape[0], 1)
        W_2 = _3_1_2_single_layer_perceptron.initializeWeights(X_2.shape[0], 1)
        W_3 = _3_1_2_single_layer_perceptron.initializeWeights(X_3.shape[0], 1)
        # print (W.shape)

        losses_0 = []
        losses_1 = []
        losses_2 = []
        losses_3 = []

        start = time.time()

        #train in batch mode
        for i in range(2):
           for _ in range(ndata):

               Y_0 = _3_1_2_single_layer_perceptron.forwardPass(W_0, X_0)
               Y_1 = _3_1_2_single_layer_perceptron.forwardPass(W_1, X_1)
               Y_2 = _3_1_2_single_layer_perceptron.forwardPass(W_2, X_2)
               Y_3 = _3_1_2_single_layer_perceptron.forwardPass(W_3, X_3)

               error_0 = _3_1_2_single_layer_perceptron.error(T_0,Y_0)
               error_1 = _3_1_2_single_layer_perceptron.error(T_1,Y_1)
               error_2 = _3_1_2_single_layer_perceptron.error(T_2,Y_2)
               error_3 = _3_1_2_single_layer_perceptron.error(T_3,Y_3)



               #print("part1.error: ", _, " ", error)
               losses_0.append(error_0)
               losses_1.append(error_1)
               losses_2.append(error_2)
               losses_3.append(error_3)

               dW_0 = -eta * np.matmul((Y_0 - T_0), np.transpose(X_0))
               dW_1 = -eta * np.matmul((Y_1 - T_1), np.transpose(X_1))
               dW_2 = -eta * np.matmul((Y_2 - T_2), np.transpose(X_2))
               dW_3 = -eta * np.matmul((Y_3 - T_3), np.transpose(X_3))

               W_0 = W_0 + dW_0
               W_1 = W_1 + dW_1
               W_2 = W_2 + dW_2
               W_3 = W_3 + dW_3


               #
               def line(x, normal):
                   # 0 = w0x + w1y + w2
                   # y = (-w0x - w2) / w1
                   return (-normal[:, 0] * x - normal[:, 2]) / normal[:, 1]


               X = np.hstack([X_0,X_1,X_2,X_3])

               maxX = X[:, np.argmax(X[0])][0] + 0.5
               minX = X[:, np.argmin(X[0])][0] - 0.5
               maxY = X[:, np.argmax(X[1])][1] + 0.5
               minY = X[:, np.argmin(X[1])][1] - 0.5

               __x = np.arange(minX, maxX, 0.05)


               #        plt.plot(x_1, line(x_1, W_1))
               #       plt.plot(x_2, line(x_2, W_2))
               #      plt.plot(x_3, line(x_3, W_3))

               def plot():
                   plt.clf()
                   plt.ylim(minY, maxY)
                   plt.xlim(minX, maxX)
                   plt.plot(A[0], A[1], "ro", B[0], B[1], "bo")
                   plt.plot(__x, _3_1_2_single_layer_perceptron.decisionBoundary(__x, W_0), label="25 each")
                   plt.plot(__x, _3_1_2_single_layer_perceptron.decisionBoundary(__x, W_1), label="50 A")
                   plt.plot(__x, _3_1_2_single_layer_perceptron.decisionBoundary(__x, W_2), label="50 B")
                   plt.plot(__x, _3_1_2_single_layer_perceptron.decisionBoundary(__x, W_3), label="splitted A")
                   plt.title("Grid")
                   plt.xlabel("X-Coord")
                   plt.ylabel("Y-Coord")
                   plt.legend()
                   plt.show()
                   plt.pause(0.000001)


               plot()

        elapsed = time.time() - start

        print(f"\nElapsed: {elapsed}")

        plt.ioff()
        plot()

    return losses_0, losses_1, losses_2, losses_3


if __name__ == "__main__":
    # print ("Mean Epochs:", np.mean([len(main()) for i in range(500)]))
    losses_0, losses_1, losses_2, losses_3 = main()
    import utility


    utility.plotLearningCurves((["25 each", "50 A", "50 B", "splitted A"], losses_0, losses_1, losses_2, losses_3))

