import numpy as np
import matplotlib.pyplot as plt
import _3_1_1_lin_seperable_data
import _3_1_2_single_layer_perceptron
import utility


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

if __name__ == "__main__":

    eta = 0.001
    ndata = 100
    subsampleCase = 3
    part = 2

    # 3.1.3 first part (Overlapping clouds)

    if part == 1:
        mA, sigmaA = [1.0, 0.3], 0.2
        mB, sigmaB = [0.5, 0.0], 0.3

        A = _3_1_1_lin_seperable_data.generateData(2, mA, sigmaA, ndata)
        B = _3_1_1_lin_seperable_data.generateData(2, mB, sigmaB, ndata)

        #plt.plot(A[0],A[1],"bo")
        #plt.plot(B[0],B[1],"ro")
        #plt.show()


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

        print(A.shape)

        #A_g = greater then 0 in A[0,:]
        A_g = A[:,:50]
        A_l = A[:,50:]


        A_l_1 = A_l[:, np.random.choice(A_l.shape[1], int(round(A_l.shape[1] * 0.80)), replace=False)] #only take 80%
        A_g_1 = A_g[:, np.random.choice(A_g.shape[1], int(round(A_g.shape[1] * 0.20)), replace=False)] #only take 20%


        A_3 = np.hstack([A_l_1,A_g_1])
        B_3 = B



    for j in range(4):
        if(j == 0):
            X = np.hstack([A_0, B_0])
            T_A, T_B = np.ones(A_0.shape[1]), -1 * np.ones(B_0.shape[1])

        elif(j == 1):
            X = np.hstack([A_1, B_1])
            T_A, T_B = np.ones(A_1.shape[1]), -1 * np.ones(B_1.shape[1])
        elif(j == 2):
            X = np.hstack([A_2, B_2])
            T_A, T_B = np.ones(A_2.shape[1]), -1 * np.ones(B_2.shape[1])
        else:
            X = np.hstack([A_3, B_3])
            T_A, T_B = np.ones(A_3.shape[1]), -1 * np.ones(B_3.shape[1])




        # Add biasing term
        X = np.vstack([X, np.ones(X.shape[1])])

        # Targets
        T = np.hstack([T_A, T_B])
        W = _3_1_2_single_layer_perceptron.initializeWeights(X.shape[0], 1)
        # print (W.shape)

        losses = []

        exit()
        #todo test error against complete data

        #train in batch mode
        for i in range(2):
            for _ in range(ndata):

                Y = _3_1_2_single_layer_perceptron.forwardPass(W, X)

                error = _3_1_2_single_layer_perceptron.error(T,Y)
                print("part1.error: ", _, " ", error)
                losses.append(error)

                dW = -eta * np.matmul((Y - T), np.transpose(X))

                W = W + dW






        #plt.plot(A[0], A[1], "ro", B[0], B[1], "bo")

        def line(x, normal):
            # 0 = w0x + w1y + w2
            # y = (-w0x - w2) / w1
            return (-normal[:,0] * x - normal[:,2]) / normal[:,1]

        maxX = X[:, np.argmax(X[1])][0]
        minX = X[:, np.argmin(X[1])][0]

        x = np.arange(minX, maxX, 0.05)

        #plt.plot(x, line(x, W))

        #plt.show()

        utility.plotLearningCurve((("train ", j) ,losses))


