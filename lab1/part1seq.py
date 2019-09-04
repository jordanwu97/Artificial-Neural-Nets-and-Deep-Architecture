import numpy as np
import matplotlib.pyplot as plt


def generateData(features, mean, sigma, n):
    """
    Returns 2 sets of data A and B
    Each dataset has shape (#features, #samples)
    """

    distA = sigma * np.random.randn(features, n)

    distA[0] = distA[0] + mean[0]
    distA[1] = distA[1] + mean[1]

    return distA


def initializeWeights(input_dim, output_dim):
    """
    Returns a weight matrix with standard normal initialization
    """
    return np.random.randn(output_dim, input_dim)      

def forwardPass(W, x):
    return np.matmul(W, x)

def error(T, Y):
    return np.sum((T - Y) ** 2) / 2

def decisionBoundary(x, normal):
    # 0 = w0x + w1y + w2
    # y = (-w0x - w2) / w1
    return (-normal[:,0] * x - normal[:,2]) / normal[:,1]

def main():

    eta = 0.01

    # Training Data sets A and B
    mA, sigmaA = [1.0, 0.5], 0.5
    mB, sigmaB = [-2.0, 0.0], 0.5

    A = generateData(2, mA, sigmaA, 100)
    B = generateData(2, mB, sigmaB, 100)
    X = np.hstack([A, B])
    # Add biasing term
    X = np.vstack([X, np.ones(X.shape[1])])

    # For matplotlib window
    maxX = X[:, np.argmax(X[0])][0] + 0.5
    minX = X[:, np.argmin(X[0])][0] - 0.5
    maxY = X[:, np.argmax(X[1])][1] + 0.5
    minY = X[:, np.argmin(X[1])][1] - 0.5

    __x = np.arange(-5, 5, 0.5)

    # Targets
    T_A, T_B = np.ones(A.shape[1]), -1 * np.ones(A.shape[1])
    T = np.hstack([T_A, T_B])

    W = initializeWeights(X.shape[0], 1)

    # plt.ion()

    e_last = 0

    losses = []

    for epoch in range(100):

        for i in range(len(T)):

            X_sample = np.transpose(np.atleast_2d(X[:, i]))
            T_sample = np.atleast_2d(T[i])
        
            WX = forwardPass(W, X_sample)

            dW = -eta * np.matmul(np.atleast_2d(WX - T_sample), np.transpose(X_sample))

            W = W + dW

        # Quickly calculate total loss for use in learning curve
        WX = forwardPass(W, X)
        e = error(T, WX)
        losses.append(e)
        
        if abs(e - e_last) < 10**-6:
            break

        e_last = e
    
    return losses

if __name__ == "__main__":
        print (np.mean([len(main()) for i in range(20)]))
        # losses = main()
        # print (losses)
        # import utility
        # utility.plotLearningCurve(losses)