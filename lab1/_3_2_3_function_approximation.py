import numpy as np

import _3_2_1_two_layer_network

def gauss(x,y):
    return np.exp(-(x**2 + y**2)/10) - 0.5

if __name__ == "__main__":
    x = np.arange(-5,5,0.5)
    y = np.arange(-5,5,0.5)

    z = gauss(x,y)

    print (z)
