from _3_1 import Hopfield, sign
from _3_2 import loadImgs, showImage
import numpy as np
import matplotlib.pyplot as plt


def plotCurves(curves, xlabel, ylabel, title, save_file=None):

    plt.clf()
    for label, vals in curves.items():
        plt.plot(np.arange(len(vals)) + 1, vals, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    if save_file:
        plt.savefig(save_file)

    plt.show()


if __name__ == "__main__":
    images = loadImgs()

    # net = Hopfield()
    # net.train(images[:3])

    # print ("Energy at Attractors")
    # for i, im in enumerate(images[:3]):
    #     print (f"p{i+1} &", net.energy(im))

    # print ("Energy at Distorted")
    # for i, im in enumerate(images[9:]):
    #     print (f"p{i+10} &", net.energy(im))

    # p = []
    # for i in (9,10):
    #     net.predict_async(images[i])
    #     p.append((f"p{i+1}", net.past_energy))

    # print (dict(p))

    # plotCurves(dict(p), "Iterations", "Energy", "Iterations vs Energy", save_file="3_3_it_vs_energy.png")

    net = Hopfield()

    net.W = np.random.randn(100, 100)

    assert not np.array_equal(net.W, net.W.T)

    x = sign(np.random.randn(100))
    net.predict_async(x)

    plotCurves(
        {"Arbitrary Starting": net.past_energy},
        "Iterations",
        "Energy",
        "Iterations vs Energy (Random Weights Non-Symetric)",
        save_file="pictures/3_3_rand_weights_nonsym.png",
    )

    net.W = 0.5 * (net.W + net.W.T)

    assert np.array_equal(net.W, net.W.T)

    x = sign(np.random.randn(100))
    net.predict_async(x)

    plotCurves(
        {"Arbitrary Starting": net.past_energy},
        "Iterations",
        "Energy",
        "Iterations vs Energy (Random Weights Symetric)",
        save_file="pictures/3_3_rand_weights_sym.png",
    )



