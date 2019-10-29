import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_squared_error, mean_absolute_error

sign = lambda x: np.where(x >= 0, 1, -1)

X = np.arange(0, 2 * np.pi, 0.1)[:,np.newaxis]
Y_SIN = np.sin(2 * X)
Y_SQUARE = sign(np.sin(2 * X))

X_VAL = np.arange(0.05, 2 * np.pi, 0.1)[:,np.newaxis]
Y_SIN_VAL = np.sin(2 * X_VAL)
Y_SQUARE_VAL = sign(np.sin(2 * X_VAL))


def repeat2D(A, rep):
    return np.repeat(A, rep).reshape(len(A), rep)


class RBF_NET:
    def __init__(self, rbfs_mean, rbfs_variance, activation=lambda x: x):


        self.rbfs_mean = rbfs_mean if rbfs_mean.ndim > 1 else rbfs_mean[:, np.newaxis]

        self.rbfs_variance = rbfs_variance
        self.activation = activation

    def phi(self, X):
        X = X if len(X.shape) == 2 else X.reshape(-1,1)
        Y = self.rbfs_mean if len(self.rbfs_mean.shape) == 2 else self.rbfs_mean.reshape(-1,1)

        phi = rbf_kernel(X, Y, gamma=1/(2 * self.rbfs_variance ** 2))

        return phi

    def train_batch(self, X, Y, max_epochs=1, eta=1, callback=None):
        """
        Return # epochs to convergence (always 1)
        """

        phi = self.phi(X)

        print (phi.shape)

        self.W = np.linalg.inv(phi.T @ phi) @ phi.T @ Y

        print (self.W.shape)
        callback() if callback else None
        return 1

    def train_delta_single(self, phi_X, Y_example, eta):
        pred = self.activation(phi_X @ self.W)
        e = (Y_example - pred)
        delW = (eta * e.T @ phi_X).T
        self.W += delW
        return np.max(np.abs(delW))

    def train_delta_batch(self, X, Y, max_epochs=1000, eta=0.01, callback=None):
        """
        Return # epochs to convergence (when error < 0.01)
        """
        self.W = np.random.randn(*self.rbfs_mean.shape)
        
        old_e = 0

        phi_X = self.phi(X)
        
        for epoch in range(1, max_epochs):
            maxDelW = 0
            for i in range(len(X)):
                Y_example = Y[i : i + 1]
                phi_X_example = phi_X[i : i + 1]
                maxDelW = max(self.train_delta_single(phi_X_example, Y_example, eta), maxDelW)
            new_e = self.mse(X,Y)
            if abs(old_e - new_e) < 10**-5:
                break
            old_e = new_e
        
        return epoch

    def predict(self, X):
        return self.activation(self.phi(X) @ self.W)

    def mse(self, X, Y):
        return mean_squared_error(self.predict(X), Y)

    def mae(self, X, Y):
        return mean_absolute_error(self.predict(X), Y)


def runWithParams(x, y, x_val, y_val, means, variance, training_method):
    """
    Returns validation error based on training with X, Y
    """
    n = RBF_NET(means, variance)
    n.__getattribute__(training_method)(x, y, callback=None)
    return n.mae(x_val, y_val)


if __name__ == "__main__":

    thresholds = set([0.1, 0.01, 0.001])

    print("Sin")
    print("Threshold & Error Achieved & Hidden \\\\")
    for hidden_num in range(1, 50):
        rbfs_mean = np.arange(0, 2 * np.pi, 2 * np.pi / hidden_num)
        rbfs_variance = 0.3
        n = RBF_NET(rbfs_mean, rbfs_variance)
        n.train_batch(X, Y_SIN)
        e = n.mae(X_VAL, Y_SIN_VAL)
        if len(thresholds) and e < max(thresholds):
            print(max(thresholds), "&", f"{e:.6f}", "&", hidden_num, "\\\\")
            thresholds.remove(max(thresholds))

        # plt.plot(X, n.predict(X))
        # plt.show(block=False)
        # plt.title(hidden_num)
        # plt.pause(.1)
        # plt.clf()


    thresholds = set([0.1, 0.01, 0.001])
    print("Square")
    print("Threshold & Error Achieved & Hidden \\\\")
    for hidden_num in range(1, 40):
        rbfs_mean = np.arange(0, 2 * np.pi, 2 * np.pi / hidden_num)
        rbfs_variance = 0.3
        n = RBF_NET(rbfs_mean, rbfs_variance)
        n.train_batch(X, Y_SQUARE)
        e = n.mae(X_VAL, Y_SQUARE_VAL)
        if len(thresholds) and e < max(thresholds):
            print(max(thresholds), "&", f"{e:.6f}", "&", hidden_num, "\\\\")
            thresholds.remove(max(thresholds))
    
        # plt.plot(X, n.predict(X))
        # plt.show(block=False)
        # plt.title(hidden_num)
        # plt.pause(.1)
        # plt.clf()


    thresholds = set([0.1, 0.01, 0.001])
    print("Square (With Sign Activation)")
    print("Threshold & Error Achieved & Hidden \\\\")
    for hidden_num in range(1, 40):
        rbfs_mean = np.arange(0, 2 * np.pi, 2 * np.pi / hidden_num)
        rbfs_variance = 0.3
        n = RBF_NET(rbfs_mean, rbfs_variance, activation=sign)
        n.train_batch(X, Y_SQUARE)
        e = n.mae(X_VAL, Y_SQUARE_VAL)
        if len(thresholds) and e < max(thresholds):
            print(max(thresholds), "&", e, "&", hidden_num, "\\\\")
            thresholds.remove(max(thresholds))
            # break