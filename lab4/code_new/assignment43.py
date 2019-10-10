from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet
from simple_dbn import save, load


if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
        dim=image_size, n_train=60000, n_test=10000
    )
    
    dbn: DeepBeliefNet = load("savefiles/dbn_greedy.pkl")

    dbn.recognize(test_imgs, test_lbls)