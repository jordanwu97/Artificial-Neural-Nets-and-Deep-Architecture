import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import _3_2_1_two_layer_network as _3_2_1
import _3_1_2_single_layer_perceptron
import utility

import pickle

### TUNING PARAMETERS
HIDDEN_LAYER_NODES = 15
TRAIN_SET_PERCENTAGE = 0.8
EPOCHS = 10000
ETA = 0.001
MAKE_VALIDATE_SET = True


def gauss(x, y):
    return np.exp(-(x ** 2 + y ** 2) / 10) - 0.5

def main():

    train_set_percentage = TRAIN_SET_PERCENTAGE

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
            test_error = _3_1_2_single_layer_perceptron.mse(
                target_test, y_out_test
            )
            test_loss.append(test_error)

        print(f"Epoch: {epoch} Train error: {error} Test Error: {test_error}")

        W_1 = W_1 + DELTA_W_1
        W_2 = W_2 + DELTA_W_2

        deltaMax = max(np.max(DELTA_W_1), np.max(DELTA_W_2))
        if deltaMax < 10**-5:
            break

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs=patterns[0], ys=patterns[1], zs=y_out.flatten())
    plt.show()

    return train_loss, test_loss


if __name__ == "__main__":

    ### visualize last y_out
    main()
    exit()

    def runNTimes(n):
        test_losses = []
        for _ in range(n):
            train_loss, test_loss = main()
            test_losses.append(test_loss[-1])
        print (f"""
        hidden nodes: {HIDDEN_LAYER_NODES}
        train set percent: {TRAIN_SET_PERCENTAGE} 
        test loss mean: {np.mean(test_losses)}
        test loss std: {np.std(test_losses)}
        """)
        return test_losses

    test_losses_means = []
    test_losses_stds = []

    # x_ = np.arange(1,27,2)
    # for h in x_:
    #     if h > 15:
    #         ETA = 0.0001
    #     HIDDEN_LAYER_NODES = h
    #     test_losses = runNTimes(10)
    #     test_losses_means.append(np.mean(test_losses))
    #     test_losses_stds.append(np.std(test_losses))
    
    # with open("loss_vs_hidden.pkl", "wb") as f:
    #     pickle.dump({
    #         "title": "loss vs hidden",
    #         "x_": x_,
    #         "test_losses_means": test_losses_means,
    #         "test_losses_stds": test_losses_stds
    #     }, f)

    with open("loss_vs_hidden.pkl", "rb") as f:
        data = pickle.load(f)

    # x_ = data["x_"]
    # test_losses_means = data["test_losses_means"]
    # test_losses_stds = data["test_losses_stds"]

    plt.title("Loss vs Hidden Nodes")
    plt.xlabel("# Hidden Nodes")
    plt.ylabel("Loss (MSE)")

    # x_ = np.arange(0.2,0.9,0.1)
    # for p in x_:
    #     TRAIN_SET_PERCENTAGE = p
    #     test_losses = runNTimes(10)
    #     test_losses_means.append(np.mean(test_losses))
    #     test_losses_stds.append(np.std(test_losses))

    # with open("loss_vs_percent_training.pkl", "wb") as f:
    #     pickle.dump({
    #         "title": "loss vs percent_traing",
    #         "x_": x_,
    #         "test_losses_means": test_losses_means,
    #         "test_losses_stds": test_losses_stds
    #     }, f)

    # with open("loss_vs_percent_training.pkl", "rb") as f:
    #     data = pickle.load(f)

    x_ = data["x_"]
    test_losses_means = data["test_losses_means"]
    test_losses_stds = data["test_losses_stds"]

    # plt.title("Loss vs Training Set %")
    # plt.xlabel("Percent of Data used for Training ")
    # plt.ylabel("Loss (MSE)")


    plt.plot(x_, test_losses_means, label="mean")
    plt.plot(x_, np.array(test_losses_means) + np.array(test_losses_stds), "--", label="std")
    plt.plot(x_, np.array(test_losses_means) - np.array(test_losses_stds), "--", label="std")
    plt.legend()
    plt.savefig(f"{data['title']}")
    plt.show()

    # print ("mean:", np.mean(test_loss_avg))
    # print ("std:", np.std(test_loss_avg))

    # utility.plotLearningCurve(("train", train_loss), ("test", test_loss))
