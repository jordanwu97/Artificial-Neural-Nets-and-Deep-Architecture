import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import _3_2_1_two_layer_network as _3_2_1
import _3_1_2_single_layer_perceptron
import utility

### TUNING PARAMETERS
HIDDEN_LAYER_NODES = 10
EPOCHS = 100
ETA = 0.001
MAKE_VALIDATE_SET = True


def gauss(x, y):
    return np.exp(-(x ** 2 + y ** 2) / 10) - 0.5

def main():

    train_set_percentage = 0.8

    hidden_layer_nodes = HIDDEN_LAYER_NODES

    # Data Set
    xx, yy = np.meshgrid(np.arange(-5, 5, 0.5), np.arange(-5, 5, 0.5), sparse=False)
    xx, yy = xx.flatten(), yy.flatten()
    patterns = np.vstack([xx, yy, np.ones(len(xx))])
    target = np.atleast_2d(gauss(xx, yy))

    # Subsample Train and Test sets
    if MAKE_VALIDATE_SET:
        num_samples = target.shape[1]
        args = np.arange(num_samples)
        np.random.shuffle(args)
        split_idx = int(num_samples * (train_set_percentage))
        train_args, test_args = args[:split_idx], args[split_idx:]
        patterns_test, patterns = patterns[:, test_args], patterns[:, train_args]
        target_test, target = target[:, test_args], target[:, train_args]

    W_1 = _3_1_2_single_layer_perceptron.initializeWeights(3, hidden_layer_nodes)
    W_2 = _3_1_2_single_layer_perceptron.initializeWeights(
        hidden_layer_nodes + 1, 1
    )

    layer1 = _3_2_1.Layer(
        activation=_3_2_1.v_sigmoid, d_activation=_3_2_1.v_d_sigmoid
    )
    layer2 = _3_2_1.Layer(
        activation=np.vectorize(lambda x: x), d_activation=np.vectorize(lambda x: 1)
    )

    train_loss = []
    test_loss = []

    for epoch in range(EPOCHS):

        ## Training
        h_out = layer1.forward(W_1, patterns, pad=True)
        y_out = layer2.forward(W_2, h_out, pad=False)
        error = _3_1_2_single_layer_perceptron.error(target, y_out)
        d_y_out = y_out - target
        d_y_in = layer2.d_in(d_y_out)
        DELTA_W_2 = _3_2_1.DELTA(ETA, d_y_in, h_out)
        d_h_out = layer1.d_out_hidden(d_y_in, W_2)
        # trim off last term of d_h_in since h_in was padded with bias
        d_h_in = layer1.d_in(d_h_out)[:-1]
        DELTA_W_1 = _3_2_1.DELTA(ETA, d_h_in, patterns)
        train_loss.append(error)

        ## Test
        test_error = 0
        if MAKE_VALIDATE_SET:
            h_out_test = layer1.forward(W_1, patterns_test, pad=True)
            y_out_test = layer2.forward(W_2, h_out_test, pad=False)
            test_error = _3_1_2_single_layer_perceptron.error(
                target_test, y_out_test
            )
            test_loss.append(test_error)

        print(f"Epoch: {epoch} Train error: {error} Test Error: {test_error}")

        W_1 = W_1 + DELTA_W_1
        W_2 = W_2 + DELTA_W_2

    # ax.scatter(xs=patterns[0], ys=patterns[1], zs=y_out.flatten())
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # plt.show()

    return train_loss, test_loss


if __name__ == "__main__":

    ### visualize last y_out
    test_loss_avg = []
    for i in range(10):
        train_loss, test_loss = main()
        test_loss_avg.append(test_loss[-1])

    print ("mean:", np.mean(test_loss_avg))
    print ("std:", np.std(test_loss_avg))

    # utility.plotLearningCurve(("train", train_loss), ("test", test_loss))
