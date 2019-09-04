import numpy as np
import matplotlib.pyplot as plt
import part1 as p1 


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

    # mode 0 = perceptron
    # mode 1 = delta rule
    mode = 1

    eta = 0.001
    ndata = 100

    
    # Sample selection case per 3.1.3
    # 0 = remove random 25% from Class A and B
    # 1 = remove random 50% from class A
    # 2 = remove random 50% from class B
    # 3 = {
    #        remove 20% from class A where classA[0,:] < 0
    #        remove 80% from subset of Class A where classA[1,:] >0
    #     }
    case = 0 

    # Training Data sets A and B
    mA, sigmaA = [1.0, 0.3], 0.2
    mB, sigmaB = [0.0, -0.1], 0.3

    A = generateNonlinearData(2, mA, sigmaA, ndata)
    B = p1.generateData(2, mB, sigmaB, ndata)
    
    # Subselect
    A_train, B_train, A_test, B_test = subselect(A,B,case)


    X = np.hstack([A_train, B_train])
    # Add biasing term
    X = np.vstack([X, np.ones(X.shape[1])])

    # For matplotlib window
    maxX = X[:, np.argmax(X[0])][0] + 0.5
    minX = X[:, np.argmin(X[0])][0] - 0.5
    maxY = X[:, np.argmax(X[1])][1] + 0.5
    minY = X[:, np.argmin(X[1])][1] - 0.5

    __x = np.arange(-5, 5, 0.5)

    # Targets
    T_A, T_B = np.ones(A_train.shape[1]), -1 * np.ones(A_train.shape[1])
    T = np.hstack([T_A, T_B])

    for mode in range(0,1):

        W = p1.initializeWeights(X.shape[0], 1)

        plt.ion()

        e_last = 0

        for epoch in range(500):
            
            WX = p1.forwardPass(W, X)
            Y = np.sign(WX)

            # Perceptron Learning
            if mode == 0:
                e = p1.error(T, Y)
                # print (Y)
                #print("Epoch: {epoch}\nPerceptron error: {e}")
                if e == 0:
                    # Stop when no more mistakes
                    break
                
                dW = -eta * np.matmul((Y - T), np.transpose(X))

            # Delta Learning
            if mode == 1:
                e = error(T, WX)
                #print("Epoch: {epoch}\nDelta error: {e}")
                # if abs(e - e_last) < 10**-5:
                    # break
                e_last = e
                dW = -eta * np.matmul((WX - T), np.transpose(X))
                print (dW)

            W = W + dW

            # Plot training points
            plt.clf()
            plt.ylim(minY,maxY)
            plt.xlim(minX,maxX)
            plt.plot(A_train[0], A_train[1], "ro", B_train[0], B_train[1], "bo")
            plt.plot(__x, p1.decisionBoundary(__x, W))
            plt.show()
            plt.pause(0.001)

        input()
