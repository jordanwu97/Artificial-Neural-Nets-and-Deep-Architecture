import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from _3_1 import RBF_NET, X, Y_SIN, Y_SQUARE, X_VAL, Y_SIN_VAL, Y_SQUARE_VAL


def getNoisy(input_):
    return input_ + np.sqrt(0.1) * np.random.randn(*(input_.shape))


def findBestParams(
    x,
    y,
    x_val,
    y_val,
    training_method,
    apply_noise=True,
    randomize_rbfs=False,
    max_epochs=100,
    hidden_range=np.arange(1, 41),
    var_range=np.round(np.arange(0.2, 1.2, 0.2), 2),
    eta=1,
):

    num_runs = 10

    errors = np.ndarray((num_runs, len(hidden_range), len(var_range)))
    num_epochs = np.ndarray((num_runs, len(hidden_range), len(var_range)))

    for run in range(num_runs):

        print ("run", run)

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
                        eps = n.__getattribute__(training_method)(
                            x, y, callback=None, max_epochs=max_epochs, eta=eta
                        )
                        cont = True
                    except KeyboardInterrupt:
                        exit()
                    except np.linalg.LinAlgError:
                        pass

                e = n.error(x_val, y_val)

                errors[run][i][j] = np.inf if np.isnan(e) else e
                num_epochs[run][i][j] = eps

    errors_mean = np.mean(errors, axis=0)
    errors_std = np.std(errors, axis=0)

    vmax_num = int(len(errors_mean.flatten()) * 0.2)
    vmax = np.partition(errors_mean.flatten(), vmax_num)[vmax_num]

    row, col = np.unravel_index(np.argmin(errors_mean), errors_mean.shape)
    best_params = hidden_range[row], var_range[col]
    best_error = errors_mean[(row, col)]
    best_error_std = errors_std[(row, col)]
    best_error_ep = np.mean(num_epochs[:, row, col])

    errors_mean[(row, col)] = -np.inf
    d = pd.DataFrame(errors_mean, index=hidden_range, columns=var_range)
    sn.heatmap(d, vmax=vmax, cmap="copper", robust=True)
    plt.xlabel(f"Variance (best={best_params[1]})")
    plt.ylabel(f"# Units (best={best_params[0]})")
    plt.title(
        f"Error Heatmap vs Params (Mean Over {num_runs} Runs) \nbest_error={best_error} \n(white spot is minimum) "
    )

    # print("Best Error:", best_error)
    # print("Best Parameters:", best_params)
    # print("Num Epchs:", best_error_ep)

    return best_error, best_error_std, best_params, best_error_ep


if __name__ == "__main__":

    # 3.2.1 3.2.3
    dat = findBestParams(X, Y_SIN, X_VAL, Y_SIN_VAL, "train_batch")
    plt.savefig("pictures/3_2_parameter_map_batch_sin.png", bbox_inches='tight')
    plt.clf()
    dat = findBestParams(X, Y_SQUARE, X_VAL, Y_SQUARE_VAL, "train_batch")
    plt.savefig("pictures/3_2_parameter_map_batch_square.png", bbox_inches='tight')
    plt.clf()

    exit()

    # dat = findBestParams(X, Y_SIN, X_VAL, Y_SIN_VAL, "train_delta_batch")
    # plt.savefig("pictures/3_2_parameter_map_delta_sin.png", bbox_inches="tight")
    # plt.clf()
    # dat = findBestParams(X, Y_SQUARE, X_VAL, Y_SQUARE_VAL, "train_delta_batch")
    # plt.savefig("pictures/3_2_parameter_map_delta_square.png", bbox_inches="tight")
    # plt.clf()

    # 3.2.2 Use best params to find rate of convergence
    # etas = [0.001,0.005,0.01,0.05,0.1,0.5,1]
    # errors = []
    # epss = []
    # for eta in etas:
    #     error,_,_,eps = findBestParams(X, Y_SIN, X_VAL, Y_SIN_VAL, "train_delta_batch", hidden_range=[12], var_range=[0.2], eta=eta)
    #     plt.clf()
    #     errors.append(error)
    #     epss.append(eps)

    # # plt.clf()
    # plt.plot(etas, epss)
    # plt.plot(etas, epss, "ro")
    # plt.xscale("log")
    # plt.ylabel("Epochs (Max=100)")
    # plt.xlabel("Eta")
    # for et, ep, err in zip(etas, epss, errors):
    #     plt.annotate(f'{np.round(err,2)}', xy=(et,ep))
    # plt.title("Eta vs Epochs (Annotated with Validation Error)")
    # plt.savefig("pictures/3_2_eta_v_ep.png", bbox_inches="tight")

    # 3.2.4 Random positioning of rbf nodes
    # dat = findBestParams(X, Y_SIN, X_VAL, Y_SIN_VAL, "train_batch", randomize_rbfs=True)
    # plt.savefig("pictures/3_2_parameter_map_batch_sin_random.png", bbox_inches='tight')
    # plt.clf()
    # dat = findBestParams(X, Y_SIN, X_VAL, Y_SQUARE_VAL, "train_batch", randomize_rbfs=True)
    # plt.savefig("pictures/3_2_parameter_map_batch_square_random.png", bbox_inches='tight')

    # 3.2.5 Test Against Clean Data
    # error,_,_,_ = findBestParams(X, Y_SIN, X_VAL, Y_SIN_VAL, "train_batch", apply_noise=False, hidden_range=[9], var_range=[.4])
    # print (" & ".join(["sin","batch","9",".4", str(error), "\\\\"]))
    # error,_,_,_ = findBestParams(X, Y_SIN, X_VAL, Y_SQUARE_VAL, "train_batch", apply_noise=False, hidden_range=[25], var_range=[.4])
    # print (" & ".join(["square","batch","25",".4", str(error), "\\\\"]))
    # error,_,_,_ = findBestParams(X, Y_SIN, X_VAL, Y_SIN_VAL, "train_delta_batch", apply_noise=False, hidden_range=[12], var_range=[.2])
    # print (" & ".join(["sin","delta","12",".2", str(error), "\\\\"]))
    # error,_,_,_ = findBestParams(X, Y_SIN, X_VAL, Y_SQUARE_VAL, "train_delta_batch", apply_noise=False, hidden_range=[15], var_range=[.2])
    # print (" & ".join(["square","delta","15",".2", str(error), "\\\\"]))

    # 3.2.6 VS MLP
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_squared_error

    for name, y, y_val, rbfs_units, rbfs_var in [
        ("sin", Y_SIN, Y_SIN_VAL, 9, 0.4),
        ("square", Y_SQUARE, Y_SQUARE_VAL, 25, 0.4),
    ]:

        y, y_val = getNoisy(np.array([y, y_val]))

        plt.plot(X_VAL, y_val, "--", label=f"true ({name})")

        # MLP

        N = len(X)
        n = MLPRegressor(
            hidden_layer_sizes=(10,),
            random_state=1,
            activation="logistic",
            solver="sgd",
            learning_rate_init=0.1,
            batch_size=N,
            max_iter=1000,
        )
        n.fit(X.reshape(-1, 1), y)
        pred = n.predict(X_VAL.reshape(-1, 1))
        mse = mean_squared_error(y_val, pred)
        plt.plot(X_VAL, pred, label=f"MLP (10 Hidden) ({name}) mse={mse}")

        # RBF
        n = RBF_NET(np.arange(0, 2 * np.pi, 2 * np.pi / rbfs_units), rbfs_var)
        n.train_batch(X, y)
        pred = n.predict(X_VAL)
        mse = mean_squared_error(y_val, pred)
        plt.plot(
            X_VAL,
            pred,
            label=f"RBF (units={rbfs_units},var={rbfs_var}) ({name}) mse={mse}",
        )

    plt.title("Regressor Performance\n")
    plt.legend()
    plt.show()
