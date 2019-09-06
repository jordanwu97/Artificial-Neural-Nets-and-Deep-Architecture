import numpy as np

def generateData(features, mean, sigma, n):
    """
    Returns 2 sets of data A and B
    Each dataset has shape (#features, #samples)
    """

    distA = sigma * np.random.randn(features, n)

    distA[0] = distA[0] + mean[0]
    distA[1] = distA[1] + mean[1]

    return distA