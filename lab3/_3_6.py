import numpy as np
from _3_3 import plotCurves
import matplotlib.pyplot as plt

sign = np.vectorize(lambda x: np.where(x >= 0, 1, -1))


# sign = np.sign

class sparse_Hopfield:
    def train(self, samples, activity):
        width = samples.shape[1]
        self.W = np.zeros((width, width))
        for x in samples:
            self.W += np.outer(x-activity, x-activity)

        # Do we get rid of diagonal?
        np.fill_diagonal(self.W,0)

        self.W = self.W / len(samples)

    def predict_sync(self, x, bias, max_iter=200):

        # self.past_energy = []
        x_cur = np.copy(x.T)

        for _ in range(max_iter):
            # self.past_energy.append(self.energy(x_cur))
            x_next = 0.5 + 0.5*sign((self.W @ x_cur) - bias)
            if np.all(x_next == x_cur):
                break
            x_cur = x_next
        
        return x_cur.T.astype(int)


if __name__ == "__main__":

    N=100
    samples = 200
    net = sparse_Hopfield()

    #how many patterns can be stored in network?

    def max_trained_patterns(b_in, activity):
        max_trained_patterns = 0
        for i in range(samples):
            
            targets = X[:i+1]

            net.train(targets, activity)
            pred = net.predict_sync(targets,b_in)
            # print (np.mean(pred))
            if np.all(pred[i] == targets[i]):
                #print(i, pred[i], X[i])
                max_trained_patterns = i+1
            else:
                break
        return max_trained_patterns

    biases = np.arange(0,2.01,0.05)

    for activity in (0.1, 0.05, 0.01):

        X = np.zeros([samples,N],dtype=int)
        for index in range(samples):
            args = np.random.choice(N, int(activity * N), replace=False)
            X[index, args] = X[index, args] + 1
    
        trained = []
        for bias in biases:
            n = max_trained_patterns(bias, activity)
            # print("max patterns: (bias ", bias, ") ", n)
            trained.append(n)

        plt.plot(biases, trained, label=f"activity={activity}")

    plt.xlabel("Bias")
    plt.ylabel("Store Images")
    plt.title("Store Images vs Bias (various activity)")
    plt.legend()
    plt.savefig("pictures/3_6_perf.png")
    plt.show()