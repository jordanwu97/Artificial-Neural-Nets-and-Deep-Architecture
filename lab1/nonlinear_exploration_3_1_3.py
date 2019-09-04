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

def subselect( A, B, case ):
    """
    Returns training and test sets as per 3.1.3 cases
    Matrix A and B's shape may not be the same due to randomly generated inputs
    """
    if case == 0:
        split = A.shape[1] *.75

        [np.random.shuffle(i) for i in A]
        [np.random.shuffle(i) for i in B]

        A_train , A_test = A[:,:int(split)], A[:,int(split):]
        B_train , B_test = B[:,:int(split)], B[:,int(split):]
    elif case == 1:
        split = A.shape[1] * .5
        A_train , A_test = A[:,:int(split)], A[:,int(split):]
        B_train , B_test = B, [] 
    elif case == 2:
        split = B.shape[1] * .5
        A_train , A_test = A , []
        B_train , B_test = B[:,:int(split)], B[:,int(split):]
    elif case == 3:

        Al = [i for i in A[0] if i < 0 ]
        Ag = [i for i in A[0] if i > 0 ]

        split1 = int(round(len(Al)*0.20))
        split2 = int(round(len(Ag)*0.80))
        Al[:split1], Al[split1:]
        Ag[:split2], Ag[split2:]
        
        A0_train, A0_test = np.concatenate((Al[:split1],Ag[:split2]),0), np.concatenate((Al[split1:],Ag[split2:]),0)
        A1_train, A1_test = A[1][:A0_train.shape[0]], A[1][A0_train.shape[0]:]

        A_train, A_test = np.asarray([A0_train, A1_train]), np.asarray([A0_test, A0_train])
        B_train, B_test = B, []

    return A_train, B_train, A_test, B_test

if __name__ == "__main__":

    eta = 0.001
    ndata = 100
    case = 3

    # Training Data sets A and B
    mA, sigmaA = [1.0, 0.3], 0.2
    mB, sigmaB = [0.0, -0.1], 0.3
    
    A = generateNonlinearData(2, mA, sigmaA, ndata)
    B = part1.generateData(2, mB, sigmaB, ndata)

    # Subselect
    A_train, B_train, A_test, B_test  = subselect( A, B, case)

    X = np.hstack([A_train, B_train])

    # Add biasing term
    X = np.vstack([X, np.ones(X.shape[1])])

    # Targets
    T_A, T_B = np.ones(A_train.shape[1]), -1 * np.ones(B_train.shape[1])
    T = np.hstack([T_A, T_B])
    W = part1.initializeWeights(X.shape[0], 1)
    # print (W.shape)

    for _ in range(ndata):

        Y = part1.forwardPass(W, X)
        #print("part1.error:", part1.error(T, Y))
        dW = -eta * np.matmul((Y - T), np.transpose(X))
        W = W + dW


    plt.plot(A_train[0], A_train[1], "ro", B_train[0], B_train[1], "bo")

    def line(x, normal):
        # 0 = w0x + w1y + w2
        # y = (-w0x - w2) / w1
        return (-normal[:,0] * x - normal[:,2]) / normal[:,1]

    maxX = X[:, np.argmax(X[1])][0]
    minX = X[:, np.argmin(X[1])][0]

    x = np.arange(minX, maxX, 0.05)

    plt.plot(x, line(x, W))

    #plt.show()
