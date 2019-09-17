import numpy as np
import matplotlib.pyplot as plt
from _4_2 import getBestValidationScore
import pickle

LAYERRANGE = range(2, 9)

# for noise_sigma in [0, 0.03, 0.09, 0.18]:
#     # val_perf_for_hidden = []
#     # for second_hidden in LAYERRANGE:
#     #     val_perf_for_hidden.append(
#     #         np.array(
#     #             [
#     #                 getBestValidationScore(
#     #                     (2, second_hidden),
#     #                     0.05,
#     #                     noise_sigma=noise_sigma,
#     #                     learn_rate=0.1,
#     #                 )
#     #                 for _ in range(20)
#     #             ]
#     #         )
#     #     )
#     #
#     # with open(f"pickles/{noise_sigma}_hidden.pkl", "wb") as f:
#     #     pickle.dump((noise_sigma, val_perf_for_hidden), f)

#     # mean = np.mean(val_perf_for_hidden, axis=1)
#     # plt.plot(LAYERRANGE, mean, label=f"sigma={noise_sigma}")

#     alphas = [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5]

#     # val_perf_regularisation = []
#     # for alpha in alphas:
#     #     val_perf_regularisation.append(
#     #         np.array(
#     #             [
#     #                 getBestValidationScore(
#     #                     (2, 4), alpha, noise_sigma=noise_sigma, learn_rate=0.1
#     #                 )
#     #                 for _ in range(10)
#     #             ]
#     #         )
#     #     )

#     # with open(f"pickles/{noise_sigma}_regularisation.pkl", "wb") as f:
#     #     pickle.dump((noise_sigma, val_perf_regularisation), f)

#     with open(f"pickles/{noise_sigma}_regularisation.pkl", "rb") as f:
#         _, val_perf_regularisation = pickle.load(f)

#     mean = np.mean(val_perf_regularisation, axis=1)
#     plt.plot(alphas, mean, label=f"sigma={noise_sigma}")

# plt.yscale("log")

# # plt.xlabel("# Hidden Nodes")
# # plt.ylabel("Validation Error (MSE)")
# # plt.title("# Hidden Nodes vs Gaussian Noise")
# # plt.legend()
# # plt.show()
# # plt.savefig("4_3_2_hidden_vs_noise.png")

# plt.xscale("log")
# plt.xlabel("Regularisation (alpha)")
# plt.ylabel("Validation Error (MSE)")
# plt.title("Regularisation vs Gaussian Noise")
# plt.legend()
# plt.savefig("pictures/4_3_2_regularisation  _vs_noise.png")
# plt.show()

alphas = [0, 0.01, 0.05, 0.1]

transform = []
for hidden in range(2,7):
    val_perf_regularisation = []
    # for alpha in alphas:
    #     val_perf_regularisation.append(
    #         np.array(
    #             [
    #                 getBestValidationScore(
    #                     (2, hidden), alpha, noise_sigma=0.09, learn_rate=0.1
    #                 )
    #                 for _ in range(10)
    #             ]
    #         )
    #     )
    # with open(f"pickles/432_{hidden}_hidden_vs_regularisation_0.09noise.pkl", "wb") as f:
    #     pickle.dump((hidden, val_perf_regularisation), f)
    with open(f"pickles/432_{hidden}_hidden_vs_regularisation_0.09noise.pkl", "rb") as f:
        _, val_perf_regularisation = pickle.load(f)

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