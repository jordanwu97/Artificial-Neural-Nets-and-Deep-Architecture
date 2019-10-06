import numpy as np
from scipy.special import expit
from util import *
from time import time
from sklearn.neural_network import BernoulliRBM
from rbm import RestrictedBoltzmannMachine


image_size = [28, 28]
train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
    dim=image_size, n_train=60000, n_test=10000
)


# s = time()
# for _ in range(10000):
#     np.dot(a,b)
# print(time() - s)

# s = time()
# for _ in range(10000):
#     a @ b
# print(time() - s)


i = BernoulliRBM(500)
u = RestrictedBoltzmannMachine(ndim_hidden=500, ndim_visible=784)

i.components_ = np.copy(u.weight_vh).T
i.intercept_hidden_ = np.copy(u.bias_h)
i.intercept_visible_ = np.copy(u.bias_v)

print(np.all(i._mean_hiddens(train_imgs) == u.get_h_given_v(train_imgs)[0]))
