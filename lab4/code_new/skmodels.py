from util import *
from sklearn.neural_network import BernoulliRBM
from rbm import RestrictedBoltzmannMachine

image_size = [28, 28]
train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
    dim=image_size, n_train=60000, n_test=10000
)

model = BernoulliRBM(
    n_components=500, learning_rate=0.01, batch_size=20, verbose=1, n_iter=10
)

self = RestrictedBoltzmannMachine(
    ndim_visible=image_size[0] * image_size[1],
    ndim_hidden=500,
    is_bottom=True,
    image_size=image_size,
    is_top=False,
    n_labels=10,
    batch_size=200,
)

model.fit(train_imgs)

h = sample_binary(sigmoid(train_imgs @ model.components_.T + model.intercept_hidden_))
p_v = sigmoid(h @ model.components_ + model.intercept_visible_)

print (np.linalg.norm(train_imgs - p_v))