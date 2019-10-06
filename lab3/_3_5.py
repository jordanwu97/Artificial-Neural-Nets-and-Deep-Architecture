from _3_1 import Hopfield, sign
from _3_2 import loadImgs, showImage
from _3_3 import plotCurves
import numpy as np
import matplotlib.pyplot as plt

import pickle

if __name__ == "__main__":

    images = loadImgs()[:9]
    images = np.flip(images, axis=0)

    def predictStableSamples(_images, noiseLevel):

        N = _images.shape[1]
        performance = []

        stable_images = []

        for train_idx in range(len(_images)):

            target = _images[: train_idx + 1]

            net = Hopfield()
            net.train(target)

            # last attempt still stable?
            sum_stable = 0
            if train_idx > 0:
                pred = net.predict_sync(stable_images, max_iter=1)
                for index in range(train_idx):
                    if net.energy(pred[index]) == net.energy(stable_images[index]):
                        sum_stable += 1
                sum_stable = sum_stable / (train_idx)
                performance.append(sum_stable)

            # calculate new stable images
            # add noise
            args = np.random.choice(N, int(noiseLevel * N), replace=False)
            noisy = np.copy(target)
            noisy[:, args] = noisy[:, args] * -1

            stable_images = net.predict_sync(noisy)
            passed = np.all(stable_images == target)

            print(train_idx, "sum_stable", sum_stable)  # , "passed: ", passed)

        return performance

    def checkStable(_images, noiseLevel, delete_diagonal=False):

        N = _images.shape[1]

        perf = []
        perf_noise = []

        for train_idx in range(0, len(_images)):

            target = _images[: train_idx + 1]
            target_noisy = np.copy(target)

            for i in range(len(target_noisy)):
                args = np.random.choice(N, int(noiseLevel * N), replace=False)
                target_noisy[i, args] = target_noisy[i, args] * -1

            net = Hopfield()
            net.train(target, delete_diagonal=delete_diagonal)

            # try reconverging the last image (target) from a noisy version of target

            if train_idx > 0:
                one_iter = net.predict_sync(target[:-1, :], max_iter=1)
                stable = np.sum(np.all(stable == one_iter, axis=1)) / len(stable)
                perf.append(stable)

                one_iter_noisy = net.predict_sync(target_noisy[:-1, :], max_iter=1)
                stable_noisy = np.sum(
                    np.all(stable_noisy == one_iter_noisy, axis=1)
                ) / len(stable_noisy)
                perf_noise.append(stable_noisy)

                print(train_idx, stable, stable_noisy)

                if np.sum(perf[-10:]) == 0:
                    break

            stable = net.predict_sync(target)
            stable_noisy = net.predict_sync(target_noisy)

        return perf, perf_noise

    def tryToTrainPredict(_images, noiseLevel, delete_diagonal=False):

        N = _images.shape[1]
        performance = []

        for train_idx in range(len(_images)):
            target = _images[: train_idx + 1]
            net = Hopfield()
            net.train(target, delete_diagonal)

            # try reconverging the last image (target) from a noisy version of target
            noisy = np.copy(target)
            for i in range(len(noisy)):
                args = np.random.choice(N, int(noiseLevel * N), replace=False)
                noisy[i, args] = noisy[i, args] * -1

            pred = net.predict_sync(noisy)

            # performance rate
            perf = np.sum(np.all(pred == target, axis=1)) / len(pred)
            performance.append(perf)

            print(train_idx + 1, perf)

            # Skip rest if performance drops to 0
            if np.sum(performance[-10:]) == 0:
                break

        return performance

    # 3.5.1
    # resultImages = tryToTrainPredict(images, 0.2)
    # resultImages0Diagonal = tryToTrainPredict(images, 0.2, delete_diagonal=True)

    # randomImages = sign(np.random.randn(200, images.shape[1]))
    # resultRandom = tryToTrainPredict(randomImages, 0.2)
    # resultRandom0Diagonal = tryToTrainPredict(randomImages, 0.2, delete_diagonal=True)

    # save = (resultImages, resultImages0Diagonal, resultRandom, resultRandom0Diagonal)
    # pickle.dump(save, open("perf.pkl", "wb"))

    (
        resultImages,
        resultImages0Diagonal,
        resultRandom,
        resultRandom0Diagonal,
    ) = pickle.load(open("perf.pkl", "rb"))

    plotCurves(
        {
            "Random Images": resultRandom,
            "Random Images (0-Diag)": resultRandom0Diagonal,
        },
        "# Images Trained",
        "Identified Correctly",
        "Performance (Given vs Random Images)",
        save_file="pictures/3_5_1_random.png",
    )

    plotCurves(
        {"Given Images": resultImages, "Given Images (0-Diag)": resultImages0Diagonal},
        "# Images Trained",
        "Identified Correctly",
        "Performance (Given vs Random Images)",
        save_file="pictures/3_5_1_given.png",
    )

    # 3.5.2
    randomImages = sign(np.random.randn(300, 100))
    # stable, stable_noise = checkStable(randomImages, 0.2)
    # stable_0d, stable_noise_0d = checkStable(randomImages, 0.2, delete_diagonal=True)

    # pickle.dump((stable_0d, stable_noise_0d), open("stability_od.pkl", "wb"))
    stable, stable_noise = pickle.load(open("stability.pkl", "rb"))
    stable_0d, stable_noise_0d = pickle.load(open("stability_od.pkl", "rb"))


    plotCurves(
        {"No Noise": stable, 
        "Noisy": stable_noise,
        "No Noise (0-Diag)": stable_0d, 
        "Noisy (0-Diag)": stable_noise_0d
        },
        "Images trained",
        "Stable",
        "Stability",
        save_file="pictures/3_5_stability.png",
    )

    # 3.5.3
    randomImages = sign(0.5 + np.random.randn(100, N))

    print(randomImages.shape)
    print("\n", "Removed self connections")
    plotCurves(
        {"Performance": tryToTrainPredict(randomImages, 0.2, delete_diagonal=True)},
        "Images trained",
        "identified correct",
        "Performance w/o Self Connection",
        save_file="pictures/3_5_performance_wo_wii_images_bias.png",
    )

    N = 100
    randomImages = sign(np.random.randn(300, N))
    print(randomImages.shape)

    print("\n", "stable after adding to Weight")
# plotCurves(
#         {"Performance": predictStableSamples(randomImages, 0.2)},
#         "Images trained",
#         "stable samples",
#         "Performance per trained Images (random samples)",
#         save_file="pictures/3_5_performance_noise_stable_samples.png",
#     )

