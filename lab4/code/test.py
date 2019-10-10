import numpy as np
import matplotlib.pyplot as plt
from util import *


if __name__ == "__main__":
    ndim_hidden = 500
    image_size = [28, 28]
    rf = {  # receptive-fields. Only applicable when visible layer is input data
        "period": 5000,  # iteration period to visualize
        "grid": [5, 5],  # size of the grid
        "ids": np.random.randint(0, ndim_hidden, 25)  # pick some random hidden units
    }

    for iteration in range(24):
        weight_vh = np.load("weights%3d.npy" %((iteration+1)))
        viz_rf(weights=weight_vh[:, rf["ids"]].reshape((image_size[0], image_size[1], -1)),
               it=iteration, grid=rf["grid"])