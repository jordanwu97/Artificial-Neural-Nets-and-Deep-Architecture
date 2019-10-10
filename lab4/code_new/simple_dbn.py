from util import *
from rbm import RestrictedBoltzmannMachine


def loadfromfile_dbn(self, loc, name, rbm):
    rbm.weight_v_to_h = np.load(
        "%s/rbm.%s.weight_v_to_h.npy" % (loc, name)
    )
    rbm.weight_h_to_v = np.load(
        "%s/rbm.%s.weight_h_to_v.npy" % (loc, name)
    )
    rbm.bias_v = np.load("%s/rbm%s.bias_v.npy" % (loc, name))
    rbm.bias_h = np.load("%s/rbm.%s.bias_h.npy" % (loc, name))
    print("loaded rbm[%s] from %s" % (name, loc))
    return


def savetofile_dbn(self, loc, name, rbm):
    np.save(
        "%s/rbm.%s.weight_v_to_h" % (loc, name), rbm.weight_v_to_h
    )
    np.save(
        "%s/rbm.%s.weight_h_to_v" % (loc, name), rbm.weight_h_to_v
    )
    np.save("%s/rbm.%s.bias_v" % (loc, name), rbm.bias_v)
    np.save("%s/rbm.%s.bias_h" % (loc, name), rbm.bias_h)
    return


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

def train_wakesleep_finetune(self, rbm0, rbm1, vis_trainset, lbl_trainset, n_iterations, name):

        """
        Wake-sleep method for learning all the parameters of network.
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("\ntraining wake-sleep..")

        try:
            raise IOError
            self.loadfromfile_dbn(loc="trained_dbn", name="0", rbm0)
            self.loadfromfile_dbn(loc="trained_dbn", name="1", rbm1)

        except IOError:

            self.n_samples = vis_trainset.shape[0]
            num_labels = lbl_trainset.shape[1]

            vis_hid = rbm0
            penlbl_top = rbm1

            accuracy = []

            for it in range(n_iterations):

                print("iteration=%7d" % it)

                for b_low in range(0, self.n_samples, self.batch_size):
                    print(b_low)
                    vis_batch = vis_trainset[b_low:b_low + self.batch_size]
                    lbl_batch = lbl_trainset[b_low:b_low + self.batch_size]

                    # vis -> wake_s_hid_h -> wake_s_pen_h / wake_s_top_v -> wake_s_top_h
                    # sleep_vis <- sleep_s_hid_h <- sleep_s_pen_h / sleep_s_top_v <- wake_s_top_h

                    # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.
                    wake_p_hid_h, wake_s_hid_h = vis_hid.get_h_given_v_dir(vis_batch)

                    wake_s_top_v = np.hstack((wake_p_hid_h, lbl_batch))
                    wake_s_top_v_0 = np.copy(wake_s_top_v)

                    wake_p_top_h, wake_s_top_h = penlbl_top.get_h_given_v(wake_s_top_v)
                    wake_s_top_h_0 = np.copy(wake_s_top_h)

                    # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.
                    for g_it in range(self.n_gibbs_wakesleep):
                        wake_p_top_v, wake_s_top_v = penlbl_top.get_v_given_h(wake_s_top_h)
                        wake_p_top_h, wake_s_top_h = penlbl_top.get_h_given_v(wake_s_top_v)

                    # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.
                    sleep_p_hid_h, sleep_s_hid_h = wake_p_top_v[:, :-num_labels], wake_s_top_v[:, :-num_labels]
                    sleep_p_vis, sleep_s_vis = vis_hid.get_v_given_h_dir(sleep_p_hid_h)

                    # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                    # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.

                    gen_p_hid_v, _ = vis_hid.get_v_given_h_dir(wake_s_hid_h)

                    rec_p_hid_h, _ = vis_hid.get_h_given_v_dir(sleep_p_vis)

                    # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.

                    vis_hid.update_generate_params(wake_s_hid_h, vis_batch, gen_p_hid_v)

                    # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.

                    penlbl_top.update_params(wake_s_top_v_0, wake_s_top_h_0, wake_p_top_v, wake_p_top_h)

                    # [TODO TASK 4.3] update generative parameters : here you will only use 'update_recognize_params' method from rbm class.

                    vis_hid.update_recognize_params(sleep_s_vis, sleep_p_hid_h, rec_p_hid_h)

                    # self.recognize(vis_batch, lbl_batch)
                accuracy.append(self.recognize(vis_trainset, lbl_trainset))

                # if it % self.print_period == 0:

        np.save(f"trained_dbn/accuracy_reco_finetune_simple_dbn_{name}", accuracy)
        plt.clf()
        plt.plot(accuracy)
        plt.show()

        self.savetofile_dbn(loc="trained_dbn", name="0", rbm0)
        self.savetofile_dbn(loc="trained_dbn", name="1", rbm1)

        return

if __name__ == "__main__":
    image_size = [28, 28]

    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
        dim=image_size, n_train=60000, n_test=10000
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


    rbm0.learning_rate = 0.05

    try:
        loadWeights(rbm0, "simple_rbm_0", "0")
    except IOError:
        rbm0.cd1(train_imgs, 10)
        saveWeights(rbm0, "simple_rbm_0", "0")

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

    rbm1.learning_rate = 0.05

    p_1_v = np.hstack((p_0_h, train_lbls))

    try:
        loadWeights(rbm1, "simple_rbm_0", "1")
    except IOError:
        rbm1.cd1(p_1_v, 10)
        saveWeights(rbm1, "simple_rbm_0", "1")

    print("recognizing...")
    recognize(rbm0, rbm1, train_imgs, train_lbls)
    recognize(rbm0, rbm1, test_imgs, test_lbls)

    print("finetuning...")
    train_wakesleep_finetune(rbm0, rbm1, vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10, name="trainset")
    train_wakesleep_finetune(rbm0, rbm1, vis_trainset=test_imgs, lbl_trainset=test_lbls, n_iterations=10, name="testset")

