from util import *
from sklearn.neural_network import BernoulliRBM
from rbm import RestrictedBoltzmannMachine

image_size = [28, 28]
train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
    dim=image_size, n_train=60000, n_test=10000
)

model = BernoulliRBM(
    n_components=500, learning_rate=0.01, batch_size=20, verbose=1, n_iter=2
)

model.fit(train_imgs)

self = RestrictedBoltzmannMachine(
    ndim_visible=image_size[0] * image_size[1],
    ndim_hidden=500,
    is_bottom=True,
    image_size=image_size,
    is_top=False,
    n_labels=10,
    batch_size=200,
)

weights = model.components_.T
bias_h = model.intercept_hidden_
bias_v = model.intercept_visible_

h = sample_binary(sigmoid(train_imgs @ weights + bias_h))

v_1 = sample_binary(sigmoid(h @ weights.T + bias_v))

print(np.sum(v_1 == train_imgs) / np.sum(v_1 == v_1))

# viz_rf(weights=weights[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=2000, grid=self.rf["grid"])
