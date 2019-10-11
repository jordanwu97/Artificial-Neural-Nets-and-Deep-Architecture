from util import *
from rbm import RestrictedBoltzmannMachine
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
        dim=image_size, n_train=60000, n_test=10000
    )

    print("\nStarting a Restricted Boltzmann Machine..")

    num_iter = 20

    for n_hidden in [200,300,400,500]:

        try:
            
            rbm: RestrictedBoltzmannMachine = pickle.load(open(f"savefiles/rbm_{n_hidden}_hidden.pkl", "rb"))
            if n_hidden == 500:
                rbm.cd1(train_imgs, 1)
        
        except IOError:

            rbm = RestrictedBoltzmannMachine(
            ndim_visible=image_size[0] * image_size[1],
            ndim_hidden=n_hidden,
            is_bottom=True,
            image_size=image_size,
            is_top=False,
            n_labels=10,
            batch_size=20,
            )

            rbm.cd1(visible_trainset=train_imgs, n_iterations=num_iter)

            pickle.dump(rbm, open(f"savefiles/rbm_{n_hidden}_hidden.pkl", "wb"))

            print (rbm.recon_losses)

        plt.plot(range(len(rbm.recon_losses)), rbm.recon_losses, label=f"num_hidden={n_hidden}")
        plt.annotate(f"{rbm.recon_losses[-1]:.5f}", (len(rbm.recon_losses) - 1,rbm.recon_losses[-1]))

    plt.xlim(-0.5, num_iter + 0.5)
    plt.title("Recon Losses vs Hidden Layers")
    plt.ylabel("Reconstruction Loss")
    plt.xlabel("# Epochs")
    plt.legend()
    plt.savefig("pictures/4_1_num_hidden.png")