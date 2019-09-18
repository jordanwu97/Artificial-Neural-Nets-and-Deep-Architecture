import numpy as np
import matplotlib.pyplot as plt
import time
import _3_1_1_lin_seperable_data

def initializeWeights(input_dim, output_dim):
    return np.random.randn(output_dim, input_dim)

def forwardPass(W, x):
    return np.matmul(W, x)

def error(T, Y):
    return np.sum((T - Y) ** 2) / 2

def mse(T, Y):
    return np.sum((T - Y) ** 2) / T.shape[1]


def decisionBoundary(x, normal):
    # 0 = w0x + w1y + w2
    # y = (-w0x - w2) / w1
    if normal.shape[1] == 3:
        return (-normal[:,0] * x - normal[:,2]) / normal[:,1]
    else:
        return (-normal[:,0] * x) / normal[:,1]

def learnPerceptron(eta, X,T,W):

    WX = forwardPass(W, X)
    Y = np.sign(WX)
    e = error(T, Y)     
    dW = -eta * np.matmul((Y - T), np.transpose(X))

    return e, dW

def learnDeltaBatch(eta, X, T, W):
    WX = forwardPass(W, X)
    e = error(T, WX)
    dW = -eta * np.matmul((WX - T), np.transpose(X))
    
    return e, dW

def learnDeltaSequential(eta, X, T, W):

    W_old = np.copy(W)

    for i in range(len(T)):

        X_sample = np.transpose(np.atleast_2d(X[:, i]))
        T_sample = np.atleast_2d(T[i])

        WX = forwardPass(W, X_sample)
        dW = -eta * np.matmul(np.atleast_2d(WX - T_sample), np.transpose(X_sample))

        W = W + dW

    WX = forwardPass(W, X)
    e = error(T, WX)
    
    return e, W-W_old


def main():

    # Training Data sets A and B
    mA, sigmaA = [2.0, 0.5], 0.5
    mB, sigmaB = [-1.0, 0.0], 0.5

    A = _3_1_1_lin_seperable_data.generateData(2, mA, sigmaA, 100)
    B = _3_1_1_lin_seperable_data.generateData(2, mB, sigmaB, 100)
    X = np.hstack([A, B])
    # Add biasing term
    X = np.vstack([X, np.ones(X.shape[1])])

    # For matplotlib window
    maxX = X[:, np.argmax(X[0])][0] + 0.5
    minX = X[:, np.argmin(X[0])][0] - 0.5
    maxY = X[:, np.argmax(X[1])][1] + 0.5
    minY = X[:, np.argmin(X[1])][1] - 0.5

    __x = np.arange(-5, 5, 0.5)

    # Targets
    T_A, T_B = np.ones(A.shape[1]), -1 * np.ones(A.shape[1])
    T = np.hstack([T_A, T_B])

    # Weights
    W_delt = initializeWeights(X.shape[0], 1)
    # remove bias
    W_delt[0][2] = 0
    W_perc = np.copy(W_delt)
    W_delt_seq = np.copy(W_delt)


    losses_perc = []
    losses_delt = []
    losses_delt_seq = []

    plt.ion()

    eta = 0.001

    convergence_threshold = 10**-6

    start = time.time()


    for epoch in range(100):
        
        # Random Initialization
        # rand = np.arange(len(T))
        # np.random.shuffle(rand)
        # X = np.transpose(np.transpose(X)[rand])
        # T = T[rand]

        e_perc, dW_perc = learnPerceptron(eta, X, T, W_perc)
        e_delt, dW_delt = learnDeltaBatch(eta, X, T, W_delt)
        #e_delt_seq, dW_delt_seq = learnDeltaSequential(eta, X, T, W_delt_seq)

       # print (dW_perc)
       # print(f"Epoch: {epoch}\nError_perc: {e_perc}")
       # print (dW_delt)
       # print(f"Epoch: {epoch}\nError_delt: {e_delt}")

        # update weights
        W_perc = W_perc + dW_perc
        W_delt = W_delt + dW_delt
       # W_delt_seq = W_delt_seq + dW_delt_seq

        #remove bias
        W_delt[0][2] = 0
        W_perc[0][2] = 0

        losses_perc.append(e_perc)
        losses_delt.append(e_delt)
       # losses_delt_seq.append(e_delt_seq)

        if np.all((np.abs(dW_delt) < convergence_threshold) & (np.abs(dW_perc) < convergence_threshold)) :
            break

        # Plot training points
        def plot():
            plt.clf()
            plt.ylim(minY,maxY)
            plt.xlim(minX,maxX)
            plt.plot(A[0], A[1], "ro", B[0], B[1], "bo")
            plt.plot(__x, decisionBoundary(__x, W_perc), label="perc")
            plt.plot(__x, decisionBoundary(__x, W_delt), label="delt")
            plt.title("Grid")
            plt.xlabel("X-Coord")
            plt.ylabel("Y-Coord")
            plt.legend()
            plt.show()
            plt.pause(0.000001)
        plot()

    elapsed = time.time() - start

    print (f"\nElapsed: {elapsed}")

    plt.ioff()
    plot()

    return losses_perc, losses_delt

if __name__ == "__main__":
    # print ("Mean Epochs:", np.mean([len(main()) for i in range(500)]))
    losses_1, losses_2 = main()
    import utility
    utility.plotLearningCurves((["train_perc", "train_delt"], losses_1, losses_2))
