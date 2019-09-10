import numpy as np

import _3_2_1_two_layer_network as _3_2_1
import _3_1_2_single_layer_perceptron

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gauss(x,y):
    return np.exp(-(x**2 + y**2)/10) - 0.5

if __name__ == "__main__":

    train_set_percentage = 0.8

    hidden_layer_nodes = 10

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Data Set
    xx, yy = np.meshgrid(np.arange(-5,5,0.5), np.arange(-5,5,0.5), sparse=False)
    xx, yy = xx.flatten(), yy.flatten()
    patterns = np.vstack([xx,yy,np.ones(len(xx))])
    target = zz = np.atleast_2d(gauss(xx,yy))

    # Subsample Train and Test sets
    num_samples = target.shape[1]
    args = np.arange(num_samples)
    np.random.shuffle(args)
    split_idx = int(num_samples*(train_set_percentage))
    train, test = args[:split_idx], args[split_idx:]

    patterns = 

    W_1 = _3_1_2_single_layer_perceptron.initializeWeights(3,hidden_layer_nodes)
    W_2 = _3_1_2_single_layer_perceptron.initializeWeights(hidden_layer_nodes + 1, 1)

    layer1 = _3_2_1.Layer(activation=_3_2_1.v_sigmoid, d_activation=_3_2_1.v_d_sigmoid)
    layer2 = _3_2_1.Layer(activation=np.vectorize(lambda x: x), d_activation=np.vectorize(lambda x: 1))


    for epoch in range(1000):

        ## Layer 1
        h_out = layer1.forward(W_1, patterns, pad=True)

        ## Layer 2
        y_out = layer2.forward(W_2, h_out, pad=False)

        print (y_out.shape)

        error = _3_1_2_single_layer_perceptron.error(target, y_out)

        print (f"Epoch: {epoch} Error: {error}")

        eta = 0.001

        d_y_out = y_out - target
        d_y_in = layer2.d_in(d_y_out)
        DELTA_W_2 = _3_2_1.DELTA(eta, d_y_in, h_out)

        d_h_out = layer1.d_out_hidden(d_y_in, W_2)
        # trim off last term of d_h_in since h_in was padded with bias
        d_h_in = layer1.d_in(d_h_out)[:-1]
        DELTA_W_1 = _3_2_1.DELTA(eta, d_h_in, patterns)

        W_1 = W_1 + DELTA_W_1
        W_2 = W_2 + DELTA_W_2


    ### visualize last y_out
    print (zz)
    print (y_out)
    ax.plot(xs=xx,ys=yy,zs=y_out.flatten())
    plt.show()


        