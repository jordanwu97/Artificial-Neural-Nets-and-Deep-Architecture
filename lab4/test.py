import numpy as np
import matplotlib.pyplot as plt

# T = 5
# a = np.vectorize(lambda x: (1 / (1 + np.exp(-x/T))))

# plt.plot(np.arange(-10,10,0.1), a(np.arange(-10,10,0.1)))
# plt.show()

p = [[0.5,0.2,0.1],[0.5,0.2,0.1]]

print (np.random.binomial(1000, p))