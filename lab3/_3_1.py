import numpy as np

sign = np.vectorize(lambda x: np.where(x>=0, 1, -1))
# sign = np.sign

class Hopfield:
    def train(self, samples):
        width = samples.shape[1]
        self.W = np.zeros((width, width))
        for x in samples:
            self.W += np.outer(x, x)

        # Do we get rid of diagonal?
        # np.fill_diagonal(self.W,0)

        self.W = self.W / len(samples)

    def predict_sync(self, x, max_iter=200):

        self.past_energy = []
        x_cur = np.copy(x.T)

        for _ in range(max_iter):
            self.past_energy.append(self.energy(x_cur))
            x_next = sign(self.W @ x_cur)
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
        for i in range(2**N):
            a = np.array(list(f"{i:b}".zfill(N)), dtype=int) * 2 - 1
            p = self.predict_sync(a)
            attractors.add(np.array2string(p))
        return attractors
    
    def energy(self, x):
        return np.round(-1 * x.T @ self.W @ x, 5)

if __name__ == "__main__":
    
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

    trained_attractors = set([ np.array2string(x) for x in X ])

    # part 1
    for x,xd in zip(X,Xd):
        p = net.predict_sync(xd)
        print (np.all(p == x))
        print (np.array2string(p) in trained_attractors)
        print (len(net.past_energy))

    # part 2 find attractors
    attractors = net.get_attractors()

    print ("Attractors:", len(attractors))
    [ print (a) for a in attractors ]
    for i, x in enumerate(X):
        print (f"x{i+1} in attractors:",np.array2string(x) in attractors)

    # part 3 more noise
    for x in X:
        rand = np.random.choice(len(x), len(x)//2, replace=False)
        xd = np.copy(x)
        xd[rand] = xd[rand] * -1
        print (x, rand, xd)
        p = net.predict_sync(xd)
        print(np.all(p == x))

