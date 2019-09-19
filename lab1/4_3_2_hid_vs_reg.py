import numpy as np
import matplotlib.pyplot as plt
from _4_2 import getBestValidationScore
import pickle

alphas = [0, 0.01, 0.05, 0.1]

transform = []
for hidden in range(2,7):
    val_perf_regularisation = []
    for alpha in alphas:
        val_perf_regularisation.append(
            np.array(
                [
                    getBestValidationScore(
                        (2, hidden), alpha, noise_sigma=0.09, learn_rate=0.1
                    )
                    for _ in range(10)
                ]
            )
        )
    with open(f"pickles/432_{hidden}_hidden_vs_regularisation_0.09noise.pkl", "wb") as f:
        pickle.dump((hidden, val_perf_regularisation), f)
    # with open(f"pickles/432_{hidden}_hidden_vs_regularisation_0.09noise.pkl", "rb") as f:
    #     _, val_perf_regularisation = pickle.load(f)

    print (np.asanyarray(val_perf_regularisation))
    mean = np.mean(val_perf_regularisation, axis=1)
    transform.append(mean)

transform = np.transpose(np.asanyarray(transform))
for reg, arr in zip(alphas, transform):
    plt.plot(range(2,7), arr.flatten(), label=f"alpha={reg}")

# print (transform)

plt.ylim((10**-3,2*10**-3))
plt.xlabel("# Hidden Nodes (2nd Layer)")
plt.yscale("log")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.title("# Hidden vs Regularisation (Sigma=0.09)")
# plt.savefig("pictures/4_3_2_hidden_vs_regularisation_0.09noise.png")
plt.show()