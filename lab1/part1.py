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
    return np.random.randn(output_dim, input_dim)

def forwardPass(W, x):
    return np.matmul(W, x)

def error(T, Y):
    return np.sum((T - Y) ** 2) / 2

if __name__ == "__main__":

    eta = 0.001

    # Training Data sets A and B
    mA, sigmaA = [1.0, 0.5], 0.5
    mB, sigmaB = [-2.0, 0.0], 0.5

    A = generateData(2, mA, sigmaA, 100)
    B = generateData(2, mB, sigmaB, 100)
    X = np.hstack([A, B])
    # Add biasing term
    X = np.vstack([X, np.ones(X.shape[1])])

    # Targets
    T_A, T_B = np.ones(A.shape[1]), -1 * np.ones(A.shape[1])
    T = np.hstack([T_A, T_B])

    W = initializeWeights(X.shape[0], 1)
    print (W.shape)


    for _ in range(100):

        Y = forwardPass(W, X)
        print("error:", error(T, Y))
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

    plt.show()
