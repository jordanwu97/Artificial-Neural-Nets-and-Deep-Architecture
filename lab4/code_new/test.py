import numpy as np
from scipy.special import expit
from util import *
from time import time
from sklearn.neural_network import BernoulliRBM
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet
import matplotlib.pyplot as plt


image_size = [28, 28]
train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
    dim=image_size, n_train=60000, n_test=10000
)

dbn = DeepBeliefNet(
        sizes={
            "vis": image_size[0] * image_size[1],
            "hid": 500,
            "pen": 500,
            "top": 2000,
            "lbl": 10,
        },
        image_size=image_size,
        n_labels=10,
        batch_size=20,
    )

dbn.train_greedylayerwise(
    vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10
)

rbm0 = dbn.rbm_stack["vis--hid"]
rbm1 = dbn.rbm_stack["hid--pen"]
rbm2 = dbn.rbm_stack["pen+lbl--top"]

image0 = train_imgs[:1,:]
lbl0 = train_lbls[:1,:]
lblany = np.ones(lbl0.shape) / 10

last = np.zeros((1,500))

losses = []
for i in range(500):
    p_0, s_0 = rbm0.get_h_given_v_dir(p_v)
    p_v, image0 = rbm0.get_v_given_h_dir(p_0)
    losses.append(np.linalg.norm(p_0-last))
    last=p_0
    if np.sum(losses[-10:]) == 0:
        break

plt.plot(range(len(losses)), losses)
plt.show()


exit()
p_1, s_1 = rbm1.get_h_given_v_dir(p_0)

random = np.random.uniform(size=(1,500))
labeled = np.hstack((random,lbl0))

for i in range(1000):
    labeled[:,-10:] = lbl0
    p_2, s_2 = rbm2.get_h_given_v(labeled)
    p_1, labeled = rbm2.get_v_given_h(s_2)

p_1, s_1 = rbm2.get_v_given_h(s_2)
p_0, s_0 = rbm1.get_v_given_h_dir(p_1[:,:-10])
vis, _ = rbm0.get_v_given_h_dir(p_0)

plt.imshow(vis.reshape(28,28))
plt.show()

# for i in range(100):
#     p_2, s_2 = rbm2.get_h_given_v(labeled)
#     p_v_2, labeled = rbm2.get_v_given_h(s_2)
#     labeled[:,:-10] = p_1
#     plt.clf()
#     plt.plot(range(10), np.log(p_v_2[0,-10:]))
#     plt.show(block=False)
#     plt.pause(0.01)