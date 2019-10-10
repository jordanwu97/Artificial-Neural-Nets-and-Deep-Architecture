from util import *
from rbm import RestrictedBoltzmannMachine


def loadWeights(rbm, dirc, name):
    rbm.weight_vh = np.load(f"{dirc}/rbm.{name}.weight_vh.npy")
    rbm.bias_v = np.load(f"{dirc}/rbm.{name}.bias_v.npy")
    rbm.bias_h = np.load(f"{dirc}/rbm.{name}.bias_h.npy")


def saveWeights(rbm: RestrictedBoltzmannMachine, dirc, name):
    np.save(f"{dirc}/rbm.{name}.weight_vh.npy", rbm.weight_vh)
    np.save(f"{dirc}/rbm.{name}.bias_v.npy", rbm.bias_v)
    np.save(f"{dirc}/rbm.{name}.bias_h.npy", rbm.bias_h)

def recognize(rbm0, rbm1, input_data, true_lbl):

    num_labels = true_lbl.shape[1]

    p_0_h, _ = rbm0.get_h_given_v_dir(input_data)
    
    s_1_v = np.hstack( (p_0_h, np.ones((p_0_h.shape[0], num_labels)) / 10) )
    
    for _ in range(20):
        s_1_v[:,:-num_labels] = p_0_h
        _, s_1_h = rbm1.get_h_given_v(s_1_v)
        _, s_1_v = rbm1.get_v_given_h(s_1_h)

    predicted_lbl = s_1_v[:,-num_labels:]

    print(
        "accuracy = %.2f%%"
        % (
            100.0
            * np.mean(
                np.argmax(predicted_lbl, axis=1) == np.argmax(true_lbl, axis=1)
            )
        )
    )

if __name__ == "__main__":
    image_size = [28, 28]

    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
        dim=image_size, n_train=6000, n_test=10000
    )

    num_labels = train_lbls.shape[1]

    rbm0 = RestrictedBoltzmannMachine(
        ndim_visible=image_size[0] * image_size[1],
        ndim_hidden=500,
        is_bottom=True,
        image_size=image_size,
        is_top=False,
        n_labels=10,
        batch_size=20,
    )

    try:
        loadWeights(rbm0, "simple_rbm", "0")
    except IOError:
        rbm0.cd1(train_imgs, 10)
        saveWeights(rbm0, "simple_rbm", "0")

    rbm0.untwine_weights()

    p_0_h, s_0_h = rbm0.get_h_given_v_dir(train_imgs)

    rbm1 = RestrictedBoltzmannMachine(
        ndim_visible=rbm0.ndim_hidden + num_labels,
        ndim_hidden=2000,
        is_bottom=False,
        image_size=image_size,
        is_top=True,
        n_labels=10,
        batch_size=20,
    )

    p_1_v = np.hstack((p_0_h, train_lbls))

    try:
        loadWeights(rbm0, "simple_rbm", "0")
    except IOError:
        rbm1.cd1(p_1_v, 10)
        saveWeights(rbm1, "simple_rbm", "1")

    recognize(rbm0, rbm1, train_imgs, train_lbls)