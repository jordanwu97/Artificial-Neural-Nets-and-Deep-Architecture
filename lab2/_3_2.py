import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from _3_1 import RBF_NET, X, Y_SIN, Y_SQUARE, X_VAL, Y_SIN_VAL, Y_SQUARE_VAL


def getNoisy(input_):
    return (input_ + 0.01 * np.random.randn(*(input_.shape)))


def findBestParams(
    x, y, x_val, y_val, training_method, apply_noise=True, randomize_rbfs=False
):

    num_runs = 20
    hidden_range = np.arange(10, 41)
    var_range = np.round(np.arange(0.05, 1.0, 0.05), 2)

    errors = np.ndarray((num_runs, len(hidden_range), len(var_range)))
    num_epochs = np.ndarray((num_runs, len(hidden_range), len(var_range)))

    for run in range(num_runs):

        print("Run", run)

        y, y_val = getNoisy(np.array([y, y_val])) if apply_noise else (y, y_val)

        for i, hidden_num in enumerate(hidden_range):
            for j, rbfs_variance in enumerate(var_range):

                cont = False
                
                while not cont:
                    rbfs_mean = (
                        np.arange(0, 2 * np.pi, 2 * np.pi / hidden_num)
                        if not randomize_rbfs
                        else np.random.uniform(0, 2 * np.pi, hidden_num)
                    )

                    n = RBF_NET(rbfs_mean, rbfs_variance)

                    try:
                        eps = n.__getattribute__(training_method)(x, y, callback=None)
                        cont = True
                    except:
                        pass

                e = n.error(x_val, y_val)

                errors[run][i][j] = np.inf if np.isnan(e) else e
                num_epochs[run][i][j] = eps

    errors_mean = np.mean(errors, axis=0)
    errors_std = np.std(errors, axis=0)

    vmax_num = 50
    vmax = np.partition(errors_mean.flatten(), vmax_num)[vmax_num]

    row, col = np.unravel_index(np.argmin(errors_mean), errors_mean.shape)
    best_params = hidden_range[row], var_range[col]
    best_error = errors_mean[(row, col)]
    best_error_std = errors_std[(row, col)]

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

    return best_error, best_error_std


if __name__ == "__main__":

    # 3.2.1 3.2.3
    # findBestParams(X, Y_SIN, X_VAL, Y_SIN_VAL, "train_batch")
    # plt.savefig("pictures/3_2_parameter_map_batch_sin.png", bbox_inches='tight')
    # plt.clf()
    # findBestParams(X, Y_SQUARE, X_VAL, Y_SQUARE_VAL, "train_batch")
    # plt.savefig("pictures/3_2_parameter_map_batch_square.png", bbox_inches='tight')
    # plt.clf()

    # findBestParams(X, Y_SIN, "train_delta_batch")
    # plt.savefig("pictures/3_2_parameter_map_delta_sin.png", bbox_inches="tight")
    # plt.clf()
    # findBestParams(X, Y_SQUARE, "train_delta_batch")
    # plt.savefig("pictures/3_2_parameter_map_delta_square.png", bbox_inches='tight')
    # plt.clf()

    # 3.2.2 Use best params to find rate of convergence

    # 3.2.4 Random positioning of rbf nodes
    print (findBestParams(X, Y_SIN, X_VAL, Y_SIN_VAL, "train_batch"))
    plt.show()
    plt.clf()
    print (findBestParams(X, Y_SIN, X_VAL, Y_SIN_VAL, "train_batch", randomize_rbfs=True))
    plt.show()
