import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from _3_1 import RBF_NET, X, Y_SIN, Y_SQUARE, X_VAL, Y_SIN_VAL, Y_SQUARE_VAL


def getNoisy(input_):
    a = input_ + 0.1 * np.random.randn(*(input_.shape))
    return a

def findBestParams(X, Y, training_method, apply_noise=False):

    num_runs = 10
    hidden_range = np.arange(10, 40)
    var_range = np.round(np.arange(0.05, 1.0, 0.05), 2)

    errors = np.ndarray((num_runs, len(hidden_range), len(var_range)))
    num_epochs = np.ndarray((num_runs, len(hidden_range), len(var_range)))

    for run in range(num_runs):

        print("Run", run)

        Y = getNoisy(np.array([Y])) if apply_noise else Y

        for i, hidden_num in enumerate(hidden_range):
            for j, rbfs_variance in enumerate(var_range):

                rbfs_mean = np.arange(0, 2 * np.pi, 2 * np.pi / hidden_num)
                n = RBF_NET(rbfs_mean, rbfs_variance)

                def cb():
                    pred = n.predict(X)
                    plt.plot(X, pred)
                    plt.plot(rbfs_mean, [0] * len(rbfs_mean), "ro")
                    plt.plot(X, Y)
                    plt.title(f"{hidden_num}, {rbfs_variance}")
                    plt.show(block=False)
                    plt.pause(0.001)
                    plt.clf()

                eps = n.__getattribute__(training_method)(X, Y, callback=None)

                e = n.error(X_VAL, Y_SIN_VAL)
                errors[run][i][j] = np.inf if np.isnan(e) else e
                num_epochs[run][i][j] = eps

    errors_mean = np.mean(errors, axis=0)

    vmax_num = 50
    vmax = np.partition(errors_mean.flatten(), vmax_num)[vmax_num]

    row, col = np.unravel_index(np.argmin(errors_mean), errors_mean.shape)
    best_params = hidden_range[row], var_range[col]
    best_error = errors_mean[(row, col)]
    errors_mean[(row, col)] = -np.inf

    d = pd.DataFrame(errors_mean, index=hidden_range, columns=var_range)
    sn.heatmap(d, vmax=vmax, cmap="copper", robust=True)
    plt.xlabel(f"Variance (best={best_params[1]})")
    plt.ylabel(f"# Units (best={best_params[0]})")
    plt.title(
        f"Error Heatmap vs Params (Mean Over {num_runs} Runs) \nbest_error={best_error} \n(white spot is minimum) "
    )

    print("Best Error:", best_error)
    print("Best Parameters:", best_params)


if __name__ == "__main__":

    # 3.2.1 3.2.3
    # findBestParams(X, Y_SIN, "train_batch")
    # plt.savefig("pictures/3_2_parameter_map_batch_sin.png", bbox_inches='tight')
    # plt.clf()
    # findBestParams(X, Y_SQUARE, "train_batch")
    # plt.savefig("pictures/3_2_parameter_map_batch_square.png", bbox_inches='tight')
    # plt.clf()

    findBestParams(X, Y_SIN, "train_delta_batch")
    plt.savefig("pictures/3_2_parameter_map_delta_sin.png", bbox_inches="tight")
    plt.clf()
    # findBestParams(X, Y_SQUARE, "train_delta_batch")
    # plt.savefig("pictures/3_2_parameter_map_delta_square.png", bbox_inches='tight')
    # plt.clf()

    # 3.2.2 Use best params to find rate of convergence

    # 3.2.4 Random positioning of rbf nodes

