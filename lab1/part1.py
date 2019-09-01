import numpy as np
import matplotlib.pyplot as plt

def generateData():
    """
    Returns 2 sets of data A and B
    Each dataset has shape (#features, #samples)
    """

    n = 100
    mA, sigmaA = [1.0, 0.5], 0.5
    mB, sigmaB = [-2.0, 0.0], 0.5

    distA = sigmaA * np.random.randn(2, n)
    distB = sigmaB * np.random.randn(2, n)

    distA[0] = distA[0] + mA[0]
    distA[1] = distA[1] + mA[1]

    distB[0] = distB[0] + mB[0]
    distB[1] = distB[1] + mB[1]

    return distA, distB

def initializeWeights():
    return np.random.randn(1, 2)

def forwardPass(W, x):
    return np.matmul(W, x)

A, B = generateData()
W = initializeWeights()
print (forwardPass(W,A))

plt.plot(A[0], A[1], "ro", B[0], B[1], "bo")
