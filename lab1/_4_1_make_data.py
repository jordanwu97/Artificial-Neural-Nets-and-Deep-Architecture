import numpy as np
import matplotlib.pyplot as plt

def generateData():
    beta = 0.2
    gamma = 0.1
    tau = 25
    n = 10

    N = 2000

    cache = np.zeros(N)
    cache[0] = 1.5

    def _x(t):
        return 0 if t < 0 else cache[t]

    def x(t):
        return _x(t-1) + (beta*_x(t-tau-1))/(1+_x(t-tau-1)**n) - gamma*_x(t-1)

    for t in range(1,N):
        cache[t] = x(t)

    return cache

__data = generateData()
__t = np.arange(301,1501)

INPUT = np.transpose(np.array([ __data[__t-offset] for offset in range(-20,1,5) ]))
OUTPUT = __data[__t+5]
OUTPUT = OUTPUT.flatten()

TRAIN_SET = INPUT[:-400], OUTPUT[:-400]
VALIDATION_SET = INPUT[-400:-200], OUTPUT[-400:-200]
TEST_SET = INPUT[-200:], OUTPUT[-200:]

if __name__ == "__main__":

    plt.plot(np.arange(len(TRAIN_SET[1])), TRAIN_SET[1], label="training+validation")
    plt.plot(np.arange(len(TRAIN_SET[1]),len(TRAIN_SET[1])+len(TEST_SET[1])),TEST_SET[1], label="testing")
    plt.show()