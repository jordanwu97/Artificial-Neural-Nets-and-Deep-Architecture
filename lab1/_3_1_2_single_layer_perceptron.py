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
    mA, sigmaA = [1.0, 0.5], 0.5
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
    W = initializeWeights(X.shape[0], 1)

    losses = []

    plt.ion()

    eta = 0.01

    convergence_threshold = 10**-6

    start = time.time()

    for epoch in range(1000):
        
        # Random Initialization
        # rand = np.arange(len(T))
        # np.random.shuffle(rand)
        # X = np.transpose(np.transpose(X)[rand])
        # T = T[rand]

        # e, dW = learnPerceptron(eta, X, T, W)
        # e, dW = learnDeltaBatch(eta, X, T, W)
        e, dW = learnDeltaSequential(eta, X, T, W)

        print (dW)

        print(f"Epoch: {epoch}\nError: {e}")
        
        # update weights
        W = W + dW

        losses.append(e)

        if np.all(np.abs(dW) < convergence_threshold):
            break

        # Plot training points
        def plot():
            plt.clf()
            plt.ylim(minY,maxY)
            plt.xlim(minX,maxX)
            plt.plot(A[0], A[1], "ro", B[0], B[1], "bo")
            plt.plot(__x, decisionBoundary(__x, W))
            plt.show()
            plt.pause(0.000001)
        plot()

    elapsed = time.time() - start

    print (f"\nElapsed: {elapsed}")

    plt.ioff()
    plot()

    return losses

if __name__ == "__main__":
    # print ("Mean Epochs:", np.mean([len(main()) for i in range(500)]))
    losses = main()
    import utility
    utility.plotLearningCurve(losses)