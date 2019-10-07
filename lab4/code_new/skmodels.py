from util import *
from sklearn.neural_network import BernoulliRBM
from rbm import RestrictedBoltzmannMachine
from sklearn.metrics import mean_squared_error

image_size = [28, 28]
train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(
    dim=image_size, n_train=6000, n_test=10000
)

n_iter = 10

v_h = BernoulliRBM(
    n_components=500, learning_rate=0.01, batch_size=20, verbose=1, n_iter=n_iter
)

try:
    v_h.components_, v_h.intercept_hidden_, v_h.intercept_visible_ = load_stuff("v_h_sklearn.npz")
except FileNotFoundError:
    v_h.fit(train_imgs)
    save_stuff("v_h_sklearn.npz", [v_h.components_, v_h.intercept_hidden_, v_h.intercept_visible_])

v_h_p_hv = sigmoid(test_imgs @ v_h.components_.T + v_h.intercept_hidden_)
v_h_p_vh = sigmoid(sample_binary(v_h_p_hv) @ v_h.components_ + v_h.intercept_visible_)


print ("Layer1 MSE:", mean_squared_error(test_imgs, v_h_p_vh))

h_p = BernoulliRBM(
    n_components=500, learning_rate=0.01, batch_size=20, verbose=1, n_iter=n_iter
)

try:
    h_p.components_, h_p.intercept_hidden_, h_p.intercept_visible_ = load_stuff("h_p_sklearn.npz")
except FileNotFoundError:
    h_p.fit(v_h_p_hv)
    save_stuff("h_p_sklearn.npz", [h_p.components_, h_p.intercept_hidden_, h_p.intercept_visible_])

h_p_p_hv = sigmoid(v_h_p_hv @ h_p.components_.T + h_p.intercept_hidden_)
h_p_p_vh = sigmoid(sample_binary(h_p_p_hv) @ h_p.components_ + h_p.intercept_visible_)

print ("Layer2 MSE:", mean_squared_error(v_h_p_hv, h_p_p_vh))

pl_top = BernoulliRBM(
    n_components=2000, learning_rate=0.01, batch_size=20, verbose=1, n_iter=n_iter
)

# data_for_top = np.hstack((h_p_p_hv, train_lbls))

try:
    pl_top.components_, pl_top.intercept_hidden_, pl_top.intercept_visible_ = load_stuff("pl_top_sklearn.npz")
except FileNotFoundError:
    pl_top.fit(data_for_top)
    save_stuff("pl_top_sklearn.npz", [pl_top.components_, pl_top.intercept_hidden_, pl_top.intercept_visible_])

testdata = np.hstack((h_p_p_hv, np.ones((len(h_p_p_hv), 10))/100))

label = np.argmax(test_lbls[:,-10:], axis=1)

p_test_v_recon = np.copy(testdata)

### TRY CLAMPING 

for i in range(20):
    p_test_hid = sigmoid(sample_binary(p_test_v_recon) @ pl_top.components_.T + pl_top.intercept_hidden_)
    p_test_v_recon = sigmoid(sample_binary(p_test_hid) @ pl_top.components_ + pl_top.intercept_visible_)
    
    # CLamp image
    p_test_v_recon[:,:-10] = testdata[:,:-10]
    
    pred_label = np.argmax(p_test_v_recon[:,-10:], axis=1)
    print (np.sum(label==pred_label)/len(pred_label))

