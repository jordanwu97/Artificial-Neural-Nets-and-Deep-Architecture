import numpy as np
import matplotlib.pyplot as plt
import part1  


def generateNonlinearData(features, mean, sigma, n):
    """
    Returns 2 sets of data A and B
    Each dataset has shape (#features, #samples)
    """

    distA = sigma * np.random.randn(features, n)

    distA[0][1:n/2] = distA[0][1:n/2] + mean[0]
    distA[0][1:n/2] = distA[0][1:n/2] - mean[0]
    distA[1] = distA[1] + mean[1]

    return distA

if __name__ == "__main__":

    eta = 0.001
    ndata = 100
    subsampleCase = 3

    # Training Data sets A and B
    mA, sigmaA = [1.0, 0.3], 0.2
    mB, sigmaB = [0.0, -0.1], 0.3

    A = generateNonlinearData(2, mA, sigmaA, ndata)
    B = part1.generateData(2, mB, sigmaB, ndata)


    # Subselection
    if subsampleCase == 0:
        A = np.asarray([np.random.choice(i, int(round(ndata * .75)), replace=False) for i in A])
        B = np.asarray([np.random.choice(i, int(round(ndata * .75)), replace=False) for i in B])
    elif subsampleCase == 1:
        A = np.asarray([np.random.choice(i, int(round(ndata * .50)), replace=False) for i in A])
    elif subsampleCase == 2:
        B = np.asarray([np.random.choice(i, int(round(ndata * .50)), replace=False) for i in B])
    elif subsampleCase == 3:

        #TODO: apply to entire Class A not A[0]
        A_l = [i for i in A[0] if i < 0 ]
        A_g = [i for i in A[0] if i > 0 ]
        A_l = np.random.choice(A_g, int(round(len(A_g) * 0.80)), replace=False)
        A_g = np.random.choice(A_l, int(round(len(A_l) * 0.20)), replace=False)
        A[1] = np.concatenate([A_l,A_g])
        print A[1]



    X = np.hstack([A, B])
    # Add biasing term
    X = np.vstack([X, np.ones(X.shape[1])])

    # Targets
    T_A, T_B = np.ones(A.shape[1]), -1 * np.ones(B.shape[1])
    T = np.hstack([T_A, T_B])
    W = part1.initializeWeights(X.shape[0], 1)
    # print (W.shape)

    for _ in range(ndata):

        Y = part1.forwardPass(W, X)
        #print("part1.error:", part1.error(T, Y))
        dW = -eta * np.matmul((Y - T), np.transpose(X))
        W = W + dW


    plt.plot(A[0], A[1], "ro", B[0], B[1], "bo")

    def line(x, normal):
        # 0 = w0x + w1y + w2
        # y = (-w0x - w2) / w1
        return (-normal[:,0] * x - normal[:,2]) / normal[:,1]

    maxX = X[:, np.argmax(X[1])][0]
    minX = X[:, np.argmin(X[1])][0]

    x = np.arange(minX, maxX, 0.05)

    plt.plot(x, line(x, W))

    #plt.show()
