from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import utility
from _4_1_make_data import TRAIN_SET, TEST_SET, VALIDATION_SET
from _3_1_2_single_layer_perceptron import mse
from sklearn.metrics import mean_squared_error

EARLY_STOP_THRESHOLD = 20 # number of fails before early stop
EARLY_STOP_TOLERANCE = 10**-6 # next validation must to better by this amount compared to this validation

pattern, target = TRAIN_SET
pattern_validation, target_validation = VALIDATION_SET
pattern_test, target_test = TEST_SET    

def getBestValidationScore(hidden_shape, regularisation):
    
    print (hidden_shape, regularisation)

    nn = MLPRegressor(
        # random_state=3,
        hidden_layer_sizes=hidden_shape,
        activation="logistic",
        solver="sgd",
        batch_size=len(target),
        learning_rate_init=0.01,
        alpha=regularisation, # L2 Penality
        early_stopping=False,
        tol=0
    )

    early_stop_count = 0
    mse_validation_list = []
    for _ in range(10000):
        nn.partial_fit(pattern, target)
        prediction_validation = nn.predict(pattern_validation)
        mse_validation_list.append(mean_squared_error(target_validation, prediction_validation))
        if _ > 2: # Do Early Stopping
            if mse_validation_list[-1] > mse_validation_list[-2] - EARLY_STOP_TOLERANCE:
                if early_stop_count > EARLY_STOP_THRESHOLD:
                    break
                else:
                    early_stop_count += 1
            else:
                early_stop_count = 0

    print ("Num Epochs:", len(nn.loss_curve_))
    print ("Validation MSE:", mse_validation_list[-1])

    ## Plot weights histogram
    # w_ = []
    # for layerWeights in nn.coefs_:
    #     w_ += list(layerWeights.flatten())
    # print (np.std(w_))
    # plt.hist(w_, bins=50, range=(-1,1))
    # plt.ylabel("Count")
    # plt.xlabel("Weights in Range")
    # plt.title(f"Histogram of Weights (Alpha={regularisation}, 50 bins)")
    # plt.savefig(f"pictures/4_3_histogram_alpha{regularisation}.png")
    # plt.clf()

    ## Plot learning curve
    # plt.plot(1 + np.arange(len(nn.loss_curve_)), nn.loss_curve_, label="Training Loss")
    # plt.plot(1 + np.arange(len(mse_validation_list)), mse_validation_list, label="Validation Loss")
    # plt.yscale("log")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss (MSE)")
    # plt.title(f"Loss VS Epochs ({hidden_shape[0]} Hidden)")
    # plt.savefig(f"pictures/4_3_loss_curve_{hidden_shape[0]}_hidden.png")
    # plt.clf()

    # Test Prediction
    prediction_test = nn.predict(pattern_test)
    print ("Test MSE:", mean_squared_error(target_test, prediction_test))
    print ("Test R2:", nn.score(pattern_test, target_test))
    return prediction_test

    return mse_validation_list[-1]

# for hidden in range(2,9,2):
#     validation = getBestValidationScore((hidden,), 0)

# Get Validation Score vs Num Hidden
# hidden_nodes = range(1,9)
# val_scores_vs_hidden = [ [ getBestValidationScore((i,), 0) for _ in range(10) ] for i in hidden_nodes ]
# mean = np.mean(val_scores_vs_hidden, axis=1)
# std = np.std(val_scores_vs_hidden, axis=1)
# plt.plot(hidden_nodes, mean, label="mean")
# plt.plot(hidden_nodes, mean+std, label="std high")
# plt.plot(hidden_nodes, mean-std, label="std low")
# plt.legend()
# plt.ylabel("Validation Score (MSE)")
# plt.xlabel("# Hidden Nodes")
# plt.title("Validation Score vs # Hidden")
# plt.show()

# # Regularisation
# regularisation = [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5]
# val_scores_vs_hidden = [ [ getBestValidationScore((6,), i) for _ in range(5) ] for i in regularisation ]
# mean = np.mean(val_scores_vs_hidden, axis=1)
# std = np.std(val_scores_vs_hidden, axis=1)
# plt.plot(regularisation, mean, label="mean")
# plt.plot(regularisation, mean+std, label="std high")
# plt.plot(regularisation, mean-std, label="std low")
# plt.legend()
# plt.ylabel("Validation Score (MSE)")
# plt.xlabel("Regularization Coefficient")
# plt.title("Validation Score vs Regularization")
# plt.xscale("log")
# plt.show()

## Plot histogram of weight
# regularisation = [0, 0.001, 0.01, 0.1, 1, 10]
# val_scores_vs_hidden = [ [ getBestValidationScore((6,), i) for _ in range(1) ] for i in regularisation ]

## Run Best Model
prediction_test = getBestValidationScore((2,), 0.05)
plt.plot(np.arange(len(prediction_test)), prediction_test, label="prediction")
plt.plot(np.arange(len(prediction_test)), target_test, label="actual")
plt.xlabel("Sample IDX")
plt.ylabel("Output Value")
plt.title("Prediction vs Actual")
plt.legend()
plt.savefig(f"pictures/4_3_prediction_vs_actual.png")