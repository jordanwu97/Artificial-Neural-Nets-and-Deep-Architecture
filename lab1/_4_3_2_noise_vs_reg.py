import numpy as np
import matplotlib.pyplot as plt
from _4_2 import getBestValidationScore
import pickle

LAYERRANGE = range(2, 9)

alphas = [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5]

for noise_sigma in [0, 0.03, 0.09, 0.18]:

    val_perf_regularisation = []
    for alpha in alphas:
        print (noise_sigma, alpha)
        val_perf_regularisation.append(
            np.array(
                [
                    getBestValidationScore(
                        (2, 4), alpha, noise_sigma=noise_sigma, learn_rate=0.1
                    )
                    for _ in range(10)
                ]
            )
        )

    with open(f"pickles/{noise_sigma}_regularisation.pkl", "wb") as f:
        pickle.dump((noise_sigma, val_perf_regularisation), f)

    with open(f"pickles/{noise_sigma}_regularisation.pkl", "rb") as f:
        _, val_perf_regularisation = pickle.load(f)

    mean = np.mean(val_perf_regularisation, axis=1)
    plt.plot(alphas[:-1], mean[:-1], label=f"sigma={noise_sigma}")

plt.xscale("log")
plt.xlabel("Regularisation (alpha)")
plt.ylabel("Validation Error (MSE)")
plt.title("Regularisation vs Gaussian Noise")
plt.legend()
plt.savefig("pictures/4_3_2_regularisation  _vs_noise.png")
plt.show()

exit()