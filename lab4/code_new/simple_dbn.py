from util import *
from rbm import RestrictedBoltzmannMachine
import pickle

def save(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def loadfromfile_dbn(loc, name, rbm):
    rbm.weight_v_to_h = np.load(
        "%s/rbm.%s.weight_v_to_h.npy" % (loc, name)
    )
    rbm.weight_h_to_v = np.load(
        "%s/rbm.%s.weight_h_to_v.npy" % (loc, name)
    )
    rbm.bias_v = np.load("%s/rbm.%s.bias_v.npy" % (loc, name))
    rbm.bias_h = np.load("%s/rbm.%s.bias_h.npy" % (loc, name))
    print("loaded rbm[%s] from %s" % (name, loc))
    return


def savetofile_dbn(loc, name, rbm):
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

def generate(rbm0, rbm1, true_lbl):

        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """

        records = []

        lbl = true_lbl
        num_label = lbl.shape[1]
        n_gibbs_gener = 200
        
        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \
        # top to the bottom visible layer (replace 'vis' from random to your generated visible layer).

        top = rbm1
        hid = rbm0
        
        p_top_v = np.random.uniform(0,1,(lbl.shape[0], top.bias_v.shape[0]))
        top_v = sample_binary(p_top_v)
     
        for it in range(n_gibbs_gener):
            p_top_v[:,-num_label:] = lbl
            top_v[:,-num_label:] = lbl
     
            p_top_h, top_h = top.get_h_given_v(p_top_v)
            p_top_v, top_v = top.get_v_given_h(p_top_h)

        v = np.zeros((28,28))

        for _ in range(100):
            top_h = sample_binary(p_top_h)
            _, top_v = top.get_v_given_h(top_h)
            vis, _ = hid.get_v_given_h_dir(top_v[:,:-num_label])
            vis = np.log(vis)
            v += vis.reshape(28,28)
        
            #plt.clf()
            #plt.imshow(v)
            #plt.show(block=False)
            #plt.pause(0.001)
            #plt.show()
            # exit()
            
        return vis

def recognize(rbm0, rbm1, input_data, true_lbl):

    num_labels = true_lbl.shape[1]

    p_0_h, _ = rbm0.get_h_given_v_dir(input_data)
    
    s_1_v = np.hstack( (p_0_h, np.ones((p_0_h.shape[0], num_labels)) / 10) )



    for _ in range(20):
        s_1_v[:,:-num_labels] = p_0_h
        _, s_1_h = rbm1.get_h_given_v(s_1_v)
        _, s_1_v = rbm1.get_v_given_h(s_1_h)

    predicted_lbl = s_1_v[:,-num_labels:]

    acc = 100.0 * np.mean( np.argmax(predicted_lbl, axis=1) == np.argmax(true_lbl, axis=1) )

    print(f"acc={acc}")

    return acc

def train_wakesleep_finetune(rbm0, rbm1, test_imgs, test_lbls, vis_trainset, lbl_trainset, n_iterations, name):

        """
        Wake-sleep method for learning all the parameters of network.
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        batch_size=200
        n_gibbs_wakesleep=5

        print("\ntraining wake-sleep..")
        n_samples = vis_trainset.shape[0]
        num_labels = lbl_trainset.shape[1]

        vis_hid = rbm0
        penlbl_top = rbm1

        accuracy = []
        accuracy.append(recognize(rbm0, rbm1, test_imgs, test_lbls))

        for it in range(n_iterations):

            print("iteration=%7d" % it)

            for b_low in range(0, n_samples, batch_size):
                print(b_low)
                vis_batch = vis_trainset[b_low:b_low + batch_size]
                lbl_batch = lbl_trainset[b_low:b_low + batch_size]

                # vis -> wake_s_hid_h -> wake_s_pen_h / wake_s_top_v -> wake_s_top_h
                # sleep_vis <- sleep_s_hid_h <- sleep_s_pen_h / sleep_s_top_v <- wake_s_top_h

                # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.
                wake_p_hid_h, wake_s_hid_h = vis_hid.get_h_given_v_dir(vis_batch)

                wake_s_top_v = np.hstack((wake_p_hid_h, lbl_batch))
                wake_s_top_v_0 = np.copy(wake_s_top_v)

                wake_p_top_h, wake_s_top_h = penlbl_top.get_h_given_v(wake_s_top_v)
                wake_s_top_h_0 = np.copy(wake_s_top_h)

                # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.
                for g_it in range(n_gibbs_wakesleep):
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
            accuracy.append(recognize(rbm0, rbm1, test_imgs, test_lbls))
            print (accuracy[-1])

        return accuracy

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
        print ("loaded rbm0")
    except IOError:
        print ("training rbm0")
        rbm0.cd1(train_imgs, 10)
        saveWeights(rbm0, "simple_rbm_0", "0")

    rbm0.untwine_weights()

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

    try:
        loadWeights(rbm1, "simple_rbm_0", "1")
        print ("loaded rbm1")
    except IOError:
        print ("training rbm1")
        p_0_h, s_0_h = rbm0.get_h_given_v_dir(train_imgs)
        p_1_v = np.hstack((p_0_h, train_lbls))
        rbm1.cd1(p_1_v, 10)
        saveWeights(rbm1, "simple_rbm_0", "1")

    #save([rbm0, rbm1], "savefiles/simpledbn_greedy.pkl")

    #print("recognizing...")
    #acc = []
    #for i in range(5):
    #    acc.append(recognize(rbm0, rbm1, test_imgs, test_lbls))
    #print (np.mean(acc), np.std(acc))

    print("generating...")
    labels = np.zeros([1,10])
    labels[0,0] = 1
    vis = generate(rbm0, rbm1, labels)
    vis = vis.reshape(28,28)
    plt.clf()
    plt.subplot(4,5,1)
    plt.imshow(vis)
    plt.show(block=False)
    plt.pause(0.1)
    for it in range(1,10):
        labels[0,it] = 1
        labels[0,it-1] = 0
        print(labels)
        vis = generate(rbm0, rbm1, labels)
        vis = vis.reshape(28,28)
        plt.subplot(4,5,it+1)
        plt.imshow(vis)
        plt.show(block=False)
        plt.pause(0.1)
    plt.savefig("pictures/4_3_generation.png")





    print("finetuning...")

    #try:
    rbm0 = load("savefiles/simpledbn_rbm_0_finetune.pkl")
    rbm1 = load("savefiles/simpledbn_rbm_1_finetune.pkl")
    acc = load("savefiles/simpledbn_acc.pkl")
    #except IOError:
   #     acc = train_wakesleep_finetune(rbm0, rbm1, test_imgs, test_lbls, vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10, name="trainset")
    #    save(rbm0, "savefiles/simpledbn_rbm_0_finetune.pkl")
    #    save(rbm1, "savefiles/simpledbn_rbm_1_finetune.pkl")
    #    save(acc, "savefiles/simpledbn_acc.pkl")

    #plt.plot(range(len(acc)), acc)
    #plt.ylabel("Train Recognition Accuracy")
    #plt.xlabel("# Epochs Fine Tuning")
    #plt.title("Fine Tuning VS Train Recognition Accuracy")
    #plt.savefig("pictures/4_3_simple_dbn_acc.png")

    #print("recognizing...")
    #acc = []
    #for i in range(5):
    #    acc.append(recognize(rbm0, rbm1, test_imgs, test_lbls))
    #print (np.mean(acc), np.std(acc))
    # recognize(rbm0, rbm1, test_imgs, test_lbls)


    print("generating...")
    labels = np.zeros([1,10])
    labels[0,0] = 1
    vis = generate(rbm0, rbm1, labels)
    vis = vis.reshape(28,28)
    plt.clf()
    plt.subplot(4,5,1)
    plt.imshow(vis)
    plt.show(block=False)
    plt.pause(0.1)
    for it in range(1,10):
        labels[0,it] = 1
        labels[0,it-1] = 0
        print(labels)
        vis = generate(rbm0, rbm1, labels)
        vis = vis.reshape(28,28)
        plt.subplot(4,5,it+1)
        plt.imshow(vis)
        plt.show(block=False)
        plt.pause(0.1)
    plt.savefig("pictures/4_3_fine_generation.png")
    

