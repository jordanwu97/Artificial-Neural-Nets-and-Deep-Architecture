import numpy as np
from _3_3 import plotCurves

sign = np.vectorize(lambda x: np.where(x >= 0, 1, -1))


# sign = np.sign

class sparse_Hopfield:
    def train(self, samples, activity):
        width = samples.shape[1]
        self.W = np.zeros((width, width))
        for x in samples:
            self.W += np.outer(x-activity, x-activity)

        # Do we get rid of diagonal?
        # np.fill_diagonal(self.W,0)

        self.W = self.W / len(samples)

    def predict_sync(self, x, bias, max_iter=200):

        self.past_energy = []
        x_cur = np.copy(x.T)

        for _ in range(max_iter):
            self.past_energy.append(self.energy(x_cur))
            x_next = 0.5 + 0.5*sign((self.W @ x_cur) - bias)
            if np.all(x_next == x_cur):
                break
            x_cur = x_next
        return x_cur.T.astype(int)

    def predict_async(self, x, max_iter=100000):

        same_energy_count = 0

        self.past_energy = []

        x_cur = np.copy(x.T)
        for _ in range(max_iter):
            idx = np.random.choice(self.W.shape[0])
            x_cur[idx] = np.sign(np.dot(self.W[idx], x_cur))
            self.past_energy.append(self.energy(x_cur))
            if _ > 1 and self.past_energy[-2] == self.past_energy[-1]:
                if same_energy_count >= 5000:
                    break
                else:
                    same_energy_count += 1
            else:
                same_energy_count = 0

        return x_cur

    def get_attractors(self):
        attractors = set()
        N = self.W.shape[0]
        for i in range(2 ** N):
            a = np.array(list(f"{i:b}".zfill(N)), dtype=int) * 2 - 1
            p = self.predict_sync(a)
            attractors.add(np.array2string(p))
        return attractors

    def energy(self, x):
        return np.round(-1 * x.T @ self.W @ x, 5)


if __name__ == "__main__":

    N=100
    samples = 100
    activity=0.1
    bias = 0
    net = sparse_Hopfield()

    #create sparse pattern
    X = np.zeros([samples,N],dtype=int)
    for index in range(samples):
        args = np.random.choice(N, int(activity * N), replace=False)
        X[index, args] = X[index, args] + 1

    #how many patterns can be stored in network?

    def max_trained_patterns(b_in):
        max_trained_patterns = 0
        for i in range(samples):
            net.train(X[:i+1], activity)
            pred = net.predict_sync(X[:i+1],b_in)
            if np.all(pred[i] == X[i]):
                #print(i, pred[i], X[i])
                max_trained_patterns = i+1
            else:
                break
        return max_trained_patterns


    trained = []
    for bias in range(20+1):
        n = max_trained_patterns(bias*0.05)
        print("max patterns: (bias ", bias*0.05, ") ", n)
        trained.append(n)

    plotCurves(
            {"Performance": trained},
            "bias * 0.05",
            "max Images trained",
            "Performance vs bias (random samples)",
            save_file="pictures/3_6_performance_patterns_10.png",
        )

    activity = 0.05
    trained = []
    for bias in range(20+1):
        n = max_trained_patterns(bias*0.05)
        print("max patterns: (bias ", bias*0.05, ") ", n)
        trained.append(n)

    plotCurves(
            {"Performance": trained},
            "bias * 0.05",
            "max Images trained",
            "Performance vs bias (random samples)",
            save_file="pictures/3_6_performance_patterns_5.png",
        )

    activity = 0.01
    trained = []
    for bias in range(20+1):
        n = max_trained_patterns(bias*0.05)
        print("max patterns: (bias ", bias*0.05, ") ", n)
        trained.append(n)

    plotCurves(
            {"Performance": trained},
            "bias * 0.05",
            "max Images trained",
            "Performance vs bias (random samples)",
            save_file="pictures/3_6_performance_patterns_1.png",
        )
