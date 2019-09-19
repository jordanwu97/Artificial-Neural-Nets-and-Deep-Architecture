import numpy as np
import matplotlib.pyplot as plt
from _4_2 import getBestValidationScore
import pickle

LAYERRANGE = range(2, 9)

for noise_sigma in [0, 0.03, 0.09, 0.18]:
    val_perf_for_hidden = []
    for second_hidden in LAYERRANGE:
        val_perf_for_hidden.append(
            np.array(
                [
                    getBestValidationScore(
                        (2, second_hidden),
                        0.05,
                        noise_sigma=noise_sigma,
                        learn_rate=0.1,
                    )
                    for _ in range(10)
                ]
            )
        )
    
    with open(f"pickles/{noise_sigma}_hidden.pkl", "wb") as f:
        pickle.dump((noise_sigma, val_perf_for_hidden), f)

    with open(f"pickles/{noise_sigma}_hidden.pkl", "rb") as f:
        _, val_perf_for_hidden = pickle.load(f)

    mean = np.mean(val_perf_for_hidden, axis=1)
    plt.plot(LAYERRANGE, mean, label=f"sigma={noise_sigma}")

plt.yscale("log")
plt.xlabel("# Hidden Nodes")
plt.ylabel("Validation Error (MSE)")
plt.title("# Hidden Nodes vs Gaussian Noise")
plt.legend()
# plt.show()
plt.savefig("4_3_2_hidden_vs_noise.png")
