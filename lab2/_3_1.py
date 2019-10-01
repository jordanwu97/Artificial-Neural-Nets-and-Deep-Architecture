import numpy as np 
import matplotlib.pyplot as plt

def repeat2D(A, rep):
    return np.repeat(A, rep).reshape(len(A), rep)

class RBF_NET():
    def __init__(self, rbfs_mean, rbfs_variance, activation=lambda x: x):
        self.rbfs_mean = rbfs_mean
        self.rbfs_variance = rbfs_variance
        self.activation = activation

    def phi(self, X):
        X_ = repeat2D(X, len(self.rbfs_mean))
        mean_ = repeat2D(self.rbfs_mean, len(X_)).T

        phi = np.exp(
            -1 * ((X_ - mean_) ** 2) / (2 * self.rbfs_variance ** 2)
        )

        return phi
        
    def train_batch(self, X, Y):
        phi = self.phi(X)
        self.W = np.linalg.inv(phi.T @ phi) @ phi.T @ Y

    def predict(self, X):
        return self.activation(self.phi(X) @ self.W)

    def error(self, X,Y):
        return np.sum((self.predict(X) - Y) ** 2)

if __name__ == "__main__":

    sign = np.vectorize(lambda x: np.where(x>=0, 1, -1))

    x = np.arange(0, 2*np.pi, 0.1)
    y_sin = np.sin(2*x)
    y_square = sign(np.sin(2*x))
    plt.plot(x, y_sin)
    plt.plot(x, y_square)

    x_test = np.arange(0.05, 2*np.pi, 0.1)
    y_sin_test = np.sin(2*x_test)
    y_square_test = sign(np.sin(2*x_test))

    thresholds = set([0.1, 0.01, 0.001])

    print ("Sin")
    print ("Threshold & Error Achieved & Hidden \\\\")
    for hidden_num in range(1,50):
        rbfs_mean = np.arange(0, 2*np.pi, 2*np.pi / hidden_num)
        rbfs_variance = 0.3
        n = RBF_NET(rbfs_mean, rbfs_variance)
        n.train_batch(x, y_sin)
        e = n.error(x_test, y_sin_test)
        if len(thresholds) and e < max(thresholds):
            print (max(thresholds), "&", e, "&", hidden_num, "\\\\")
            thresholds.remove(max(thresholds))
    
    thresholds = set([0.1, 0.01, 0.001])
    print ("Square")
    print ("Threshold & Error Achieved & Hidden \\\\")
    for hidden_num in range(1,40):
        rbfs_mean = np.arange(0, 2*np.pi, 2*np.pi / hidden_num)
        rbfs_variance = 0.3
        n = RBF_NET(rbfs_mean, rbfs_variance, activation=sign)
        n.train_batch(x, y_square)
        e = n.error(x_test, y_square_test)
        print (e)
        if len(thresholds) and e < max(thresholds):
            print (max(thresholds), "&", e, "&", hidden_num, "\\\\")
            thresholds.remove(max(thresholds))
        plt.plot(x, n.predict(x))
        plt.show(block=False)
        plt.title(hidden_num)
        plt.pause(.1)
        plt.clf()