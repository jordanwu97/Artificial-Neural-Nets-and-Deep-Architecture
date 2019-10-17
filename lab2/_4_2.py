import numpy as np
from _4_1 import *
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

def loadCities():
    with open("data/cities.dat") as f:
        rows = f.read().split("\n")[4:-1]
        rows = [ row[:-1].split(", ") for row in rows ]
        rows = np.array(rows, dtype=float)
        return rows

def getFinalLen(points):
    print (points)
    points_shift = np.array(points)[np.arange(1,len(points) + 1) % len(points)]
    print (points_shift)
    pw = pairwise_distances(np.array(points), points_shift)
    return np.sum(np.diag(pw))

if __name__ == "__main__":

    datapoints = loadCities()
    labels = np.arange(len(datapoints))

    num_nodes = 10
    num_epochs = 100

    def neighbor_size(ep):
        if ep < (num_epochs * 1/3):
            return 2
        if ep < (num_epochs * 2/3):
            return 1
        return 0

    def circular_neighbor(center, neighborhood):
        low = center - neighborhood
        high = center + neighborhood
        return np.mod(np.arange(low, high+1), num_nodes)

    som = SOM(num_nodes, circular_neighbor, neighbor_size)
    som.train(datapoints, num_epochs)
    order = som.showMap(datapoints, labels)

    print (getFinalLen(datapoints[order]))

    plt.plot(datapoints.T[0], datapoints.T[1], "ro")
    plt.plot(datapoints[order].T[0], datapoints[order].T[1], "--")
    for i, point in enumerate(datapoints[order]):
        plt.annotate(i, point)
    plt.show()
    