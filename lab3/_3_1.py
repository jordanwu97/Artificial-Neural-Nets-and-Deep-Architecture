import numpy as np


class Hopfield:
    def train(self, samples):
        width = samples.shape[1]
        self.W = np.zeros((width, width))
        for x in samples:
            self.W += np.outer(x, x)

        # Do we get rid of diagonal?
        # np.fill_diagonal(self.W,0)

        # self.W = self.W / len(samples)

    def predict_sync(self, x, max_iter=200):
        x_cur = x.T
        for _ in range(max_iter):
            x_next = np.sign(self.W @ x_cur)
            if np.all(x_next == x_cur):
                break
            x_cur = x_next

        return x_cur.T.astype(int)


x1 = [-1, -1, 1, -1, 1, -1, -1, 1]
x2 = [-1, -1, -1, -1, -1, 1, -1, -1]
x3 = [-1, 1, 1, -1, -1, 1, -1, 1]
X = np.vstack((x1, x2, x3))

x1d = [1, -1, 1, -1, 1, -1, -1, 1]
x2d = [1, 1, -1, -1, -1, 1, -1, -1]
x3d = [1, 1, 1, -1, 1, 1, -1, 1]
Xd = np.vstack((x1d, x2d, x3d))

net = Hopfield()

net.train(X)

# for x in Xd:
print()
print(np.all(net.predict_sync(Xd) == X))
print(Xd == X)

for i in range(256):
    print (f"{i:08b}")

