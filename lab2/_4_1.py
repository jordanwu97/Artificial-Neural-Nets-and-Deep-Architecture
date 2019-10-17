import numpy as np
import re
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances

def loadAnimalData():
    with open("data/animalnames.txt") as f:
        names = np.array([ re.match("'(.*)'", name).group(1) for name in f.readlines() ])
    with open("data/animalattributes.txt") as f:
        attr = np.array([ attr[:-1] for attr in f.readlines() ])
    with open("data/animals.dat") as f:
        datapoints = np.array([int(a) for a in f.read().split(",")]).reshape(len(names),len(attr))
    
    return names, attr, datapoints

class SOM:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def getNeighborArgs(self, center, nb_size, maxsize):
        low = max(center - nb_size,0)
        high = min(center + nb_size, maxsize-1)
        return np.arange(low, high+1)

    def train(self, dataMatrix: np.ndarray, n_epochs=20, eta=0.2):
        
        num_features = dataMatrix.shape[1]

        nodes = np.random.uniform(low=0,high=1,size=(self.num_nodes, num_features))

        for ep in range(1,n_epochs+1):
            
            nb_size = (n_epochs - ep) + 5

            for p in dataMatrix:
                p = p.reshape(1,-1)

                center_arg, _ = pairwise_distances_argmin_min(p, nodes)
                nbs_args = self.getNeighborArgs(center_arg, nb_size, len(nodes))

                nodes[nbs_args] += eta * (p - nodes[nbs_args])

        self.nodes = nodes

    def showMap(self, dataMatrix, labels):

        label_arg = [ (label, pairwise_distances_argmin_min(p.reshape(1,-1), self.nodes)[0][0]) for label, p in zip(labels, dataMatrix) ]
        label_arg = sorted(label_arg, key=lambda x:x[1])
        print (label_arg)

if __name__ == "__main__":
    
    names, attr, datapoints = loadAnimalData()

    som = SOM(100)
    som.train(datapoints)
    som.showMap(datapoints, names)