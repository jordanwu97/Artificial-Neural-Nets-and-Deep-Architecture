import numpy as np
import matplotlib.pyplot as plt

sign = np.vectorize(lambda x: np.where(x >= 0, 1, -1))

X = np.arange(0, 2 * np.pi, 0.1)
Y_SIN = np.sin(2 * X)
Y_SQUARE = sign(np.sin(2 * X))

X_VAL = np.arange(0.05, 2 * np.pi, 0.1)
Y_SIN_VAL = np.sin(2 * X_VAL)
Y_SQUARE_VAL = sign(np.sin(2 * X_VAL))


def repeat2D(A, rep):
    return np.repeat(A, rep).reshape(len(A), rep)


class RBF_NET:
    def __init__(self, rbfs_mean, rbfs_variance, activation=lambda x: x):
        self.rbfs_mean = rbfs_mean
        self.rbfs_variance = rbfs_variance
        self.activation = activation

    def phi(self, X):
        X_ = repeat2D(X, len(self.rbfs_mean))
        mean_ = repeat2D(self.rbfs_mean, len(X_)).T

        phi = np.exp(-1 * ((X_ - mean_) ** 2) / (2 * self.rbfs_variance ** 2))

        return phi

    def train_batch(self, X, Y, max_epochs=1, eta=1, callback=None):
        """
        Return # epochs to convergence (always 1)
        """

        phi = self.phi(X)
        self.W = np.linalg.inv(phi.T @ phi) @ phi.T @ Y
        callback() if callback else None
        return 1

    def train_delta_single(self, X_scalar, Y_scalar, eta=1):
        phi_ = self.phi(X_scalar)
        pred = self.activation(phi_ @ self.W)
        e = (Y_scalar - pred)[0]
        delW = (eta * e * phi_).flatten()
        self.W += delW
        return np.max(np.abs(delW))

    def train_delta_batch(self, X, Y, max_epochs=100, eta=1, callback=None):
        """
        Return # epochs to convergence (when error < 0.01)
        """
        self.W = np.random.randn(*self.rbfs_mean.shape)
        
        old_e = 0
        
        for _ in range(1, max_epochs):
            # print (_)
            maxDelW = 0
            for i in range(len(X)):
                maxDelW = max(self.train_delta_single(X[i : i + 1], Y[i : i + 1], eta=eta), maxDelW)
            new_e = self.error(X,Y)
            if abs(old_e - new_e) < 10**-5:
                break
            old_e = new_e

        return _

    def predict(self, X):
        return self.activation(self.phi(X) @ self.W)

    def error(self, X, Y):
        return np.sum((self.predict(X) - Y) ** 2)


def runWithParams(x, y, x_val, y_val, means, variance, training_method):
    """
    Returns validation error based on training with X, Y
    """
    n = RBF_NET(means, variance)
    n.__getattribute__(training_method)(x, y, callback=None)
    return n.error(x_val, y_val)


if __name__ == "__main__":

    thresholds = set([0.1, 0.01, 0.001])

    print("Sin")
    print("Threshold & Error Achieved & Hidden \\\\")
    for hidden_num in range(1, 50):
        rbfs_mean = np.arange(0, 2 * np.pi, 2 * np.pi / hidden_num)
        rbfs_variance = 0.3
        n = RBF_NET(rbfs_mean, rbfs_variance)
        n.train_batch(X, Y_SIN)
        e = n.error(X_VAL, Y_SIN_VAL)
        if len(thresholds) and e < max(thresholds):
            print(max(thresholds), "&", e, "&", hidden_num, "\\\\")
            thresholds.remove(max(thresholds))

    thresholds = set([0.1, 0.01, 0.001])
    print("Square")
    print("Threshold & Error Achieved & Hidden \\\\")
    for hidden_num in range(1, 40):
        rbfs_mean = np.arange(0, 2 * np.pi, 2 * np.pi / hidden_num)
        rbfs_variance = 0.3
        n = RBF_NET(rbfs_mean, rbfs_variance, activation=sign)
        n.train_batch(X, Y_SQUARE)
        e = n.error(X_VAL, Y_SQUARE_VAL)
        if len(thresholds) and e < max(thresholds):
            print(max(thresholds), "&", e, "&", hidden_num, "\\\\")
            thresholds.remove(max(thresholds))
            break

        # plt.plot(X, n.predict(X))
        # plt.show(block=False)
        # plt.title(hidden_num)
        # plt.pause(.1)
        # plt.clf()
