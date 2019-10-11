from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet
from simple_dbn import save, load

if __name__ == "__main__":

    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
        dim=image_size, n_train=60000, n_test=10000
    )

    # """ restricted boltzmann machine """

    # print("\nStarting a Restricted Boltzmann Machine..")

    # for n_hidden in [500, 400, 300, 200]:

    #     rbm = RestrictedBoltzmannMachine(
    #         ndim_visible=image_size[0] * image_size[1],
    #         ndim_hidden=n_hidden,
    #         is_bottom=True,
    #         image_size=image_size,
    #         is_top=False,
    #         n_labels=10,
    #         batch_size=20,
    #     )

    #     rbm.cd1(visible_trainset=train_imgs, n_iterations=10)

    #     save_stuff(f"savefiles/{n_hidden}_hidden_trained_params.npz", [rbm.weight_vh, rbm.bias_h, rbm.bias_v])
    # exit()

    """ deep- belief net """

    print("\nStarting a Deep Belief Net..")

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
        batch_size=200,
    )

    """ greedy layer-wise training """

    dbn.train_greedylayerwise(
        vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10
    )

    # dbn.recognize(train_imgs, train_lbls)

    #for digit in range(1,10):
    #     digit_1hot = np.zeros(shape=(1, 10))
    #     digit_1hot[0, digit] = 1
    #     print (digit_1hot)
    #     dbn.generate(digit_1hot, name="rbms")
        
    """ fine-tune wake-sleep training """

    dbn: DeepBeliefNet = load("savefiles/dbn_greedy.pkl")

    dbn.recognize(test_imgs[:1000], test_lbls[:1000])

    dbn.train_wakesleep_finetune(
        vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10
    )

    from simple_dbn import save, load

    save(dbn, "savefiles/originaldbn_finetuned")

    dbn.recognize(test_imgs[:1000], test_lbls[:1000])

    # acc = np.load("trained_dbn/accuracy_reco_finetune_normal_dbn.npy")

    # plt.plot(range(len(acc)), acc)
    # plt.xlabel("Fine Tuning Epochs")
    # plt.ylabel("Train Set Accuracy")
    # plt.title("Fine Tuning vs Accuracy")
    # plt.savefig("pictures/4_3_fine_tune_acc.png")

    for name, im, lb in [("train", train_imgs, train_lbls), ("test", test_imgs, test_lbls)]:
        acc = []
        for trials in range(10):
            acc.append(dbn.recognize(train_imgs, train_lbls))
            # print (name, acc[-1])

        print (f"{name} & {np.mean(acc):.5f} & {np.std(acc):.5f}")

    # dbn.recognize(train_imgs, train_lbls)

    # dbn.recognize(test_imgs, test_lbls)

    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1, 10))
    #     digit_1hot[0, digit] = 1
    #     dbn.generate(digit_1hot, name="dbn")
