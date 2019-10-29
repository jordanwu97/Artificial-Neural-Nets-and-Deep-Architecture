import numpy as np
import re
from sklearn.metrics import pairwise_distances_argmin, pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt

def loadAnimalData():
    with open("data/animalnames.txt") as f:
        names = np.array([ re.match("'(.*)'", name).group(1) for name in f.readlines() ])
    with open("data/animalattributes.txt") as f:
        attr = np.array([ attr[:-1] for attr in f.readlines() ])
    with open("data/animals.dat") as f:
        datapoints = np.array([int(a) for a in f.read().split(",")]).reshape(len(names),len(attr))
    
    return names, attr, datapoints

class SOM:

    def __init__(self, num_nodes, get_neighbors_func, neighbor_size_func):
        self.num_nodes = num_nodes
        self.get_neighbors = get_neighbors_func
        self.neighbor_size = neighbor_size_func

    def train(self, dataMatrix: np.ndarray, n_epochs=20, eta=0.2):
        
        num_features = dataMatrix.shape[1]

        nodes = np.random.uniform(low=0,high=1,size=(self.num_nodes, num_features))

        for ep in range(1,n_epochs+1):
            
            nb_size = self.neighbor_size(ep)

            for p in dataMatrix:
                p = p.reshape(1,-1)

                center_arg = pairwise_distances_argmin(p, nodes)

                nodes += eta * (p - nodes) * self.get_neighbors(center_arg, nb_size)

        self.nodes = nodes

    def showMap(self, dataMatrix, labels):

        best_nodes = pairwise_distances_argmin(dataMatrix, self.nodes)

        self.map = best_nodes

        sorted_args = np.argsort(best_nodes)

        return labels[sorted_args]
        

if __name__ == "__main__":
    
    names, attr, datapoints = loadAnimalData()

    num_nodes = 100

    indexing = np.arange(num_nodes)[:,np.newaxis]

    def get_neighbors_linear(center, nb_size):
        
        return rbf_kernel(indexing, [center], gamma=(1/nb_size**2))

    som = SOM(num_nodes, get_neighbors_linear, lambda ep: (20 - ep) + 5)

    som.train(datapoints)
    
    order = som.showMap(datapoints, names)

    print (len(order))

    # plt.xticks(np.arange(len(order)), order)
    plt.plot(range(len(order)), range(len(order)), "o")
    for i, animal in enumerate(order):
        plt.annotate(animal, (i,i))
    plt.show()



    # plt.plot(np.arange(num_nodes), np.zeros((num_nodes,)), "o")
    # plt.show()