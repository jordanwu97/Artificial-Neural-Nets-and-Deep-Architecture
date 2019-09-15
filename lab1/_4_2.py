from sklearn.neural_network import MLPRegressor
import numpy as np
# from _3_2_3_function_approximation import makeData
from _4_1_make_data import TRAIN_SET, TEST_SET

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from _3_1_2_single_layer_perceptron import error as errorfunc

pattern, target = TRAIN_SET
target = target.flatten()

pattern_test, target_test = TEST_SET
target_test = target_test.flatten()

nn = MLPRegressor(
    random_state=2,
    hidden_layer_sizes=(18,),
    activation="logistic",
    solver="sgd",
    batch_size=len(target),
    learning_rate_init=0.1,
    alpha=0,
    max_iter=10000,
    early_stopping=True
)

nn.fit(pattern, target)


# for i in range(1000):
#     nn.partial_fit(pattern, target)

prediction_test = nn.predict(pattern_test)

print (nn.score(pattern_test, target_test))

plt.plot(np.arange(len(nn.validation_scores_)), 1 - np.array(nn.validation_scores_), label="Validation Loss")
plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(xs=np.transpose(pattern)[0], ys=np.transpose(pattern)[1], zs=y_out)
# plt.show()