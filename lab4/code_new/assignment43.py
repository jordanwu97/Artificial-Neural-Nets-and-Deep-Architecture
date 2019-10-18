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

    for digit in range(0,10):
        digit_1hot = np.zeros(shape=(1, 10))
        digit_1hot[0, digit] = 1
        vis = dbn.generate(digit_1hot)

        plt.subplot(2,5,digit+1)
        plt.axis("off")
        plt.title(f"Label={digit}")
        plt.imshow(vis)

    plt.tight_layout()
    plt.savefig("pictures/4_3_greedy_generated.png", bbox_inches="tight")
    plt.show()

    try:
        dbn: DeepBeliefNet = load("savefiles/dbn_fine_tuned.pkl")
    except IOError:
        dbn.train_wakesleep_finetune(train_imgs, train_lbls, 10)
        save(dbn, "savefiles/dbn_fine_tuned.pkl")

    ## Fine tuning is done by here

    # plt.plot(range(len(dbn.accuracy)), dbn.accuracy)
    # plt.xlabel("Epochs Fine Tuning")
    # plt.ylabel("Trainset Recognition Accuracy")
    # plt.title("Fine Tuning Performance Gain")
    # plt.savefig("pictures/4_3_dbn_fine_tune_acc.png")

    # for name, im, lb in [("train", train_imgs, train_lbls), ("test", test_imgs, test_lbls)]:
    #     acc = []
    #     for trials in range(10):
    #         acc.append(dbn.recognize(train_imgs, train_lbls))
    #         print (name, acc[-1])

    #     print (f"{name} & {np.mean(acc):.5f} & {np.std(acc):.5f}")

    for digit in range(0,10):
        digit_1hot = np.zeros(shape=(1, 10))
        digit_1hot[0, digit] = 1
        vis = dbn.generate(digit_1hot)

        plt.subplot(2,5,digit+1)
        plt.axis("off")
        plt.title(f"Label={digit}")
        plt.imshow(vis)

    plt.tight_layout()
    plt.savefig("pictures/4_3_fine_tuned_generated.png", bbox_inches="tight")
    plt.show()