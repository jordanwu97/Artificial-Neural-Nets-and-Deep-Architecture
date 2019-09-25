import matplotlib.pyplot as plt
import numpy as np
from _3_1 import Hopfield

def loadImgs():
    with open("pict.dat", "r") as f:
        dat = f.read().split(",")   
    return np.reshape(dat, (len(dat)//1024,1024)).astype(int)

if __name__ == "__main__":

    images = loadImgs()

    for dat in images[:0]:
        plt.imshow(np.reshape(dat,(32,32)))
        plt.show()

    net = Hopfield()
    net.train(images[:3])

    print (net.get_attractors())