from _3_1 import Hopfield, sign
from _3_2 import loadImgs, showImage
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    images = loadImgs()[:9]
    images = np.flip(images, axis=0)
    N = images.shape[1]

    def tryToTrainPredict(_images, noiseLevel):

        for train_idx in range(len(_images)):
            target = _images[:train_idx + 1]

            net = Hopfield()
            net.train(target)

            # try reconvering the last image (target) from a noisy version of target
            args = np.random.choice(N, int(noiseLevel * N), replace=False)
            noisy = np.copy(target)
            noisy[:,args] = noisy[:,args] * -1
            pred = net.predict_sync(noisy)

            passed = np.all(pred == target)

            # print ("Target Energy", net.energy(target[0]))
            # print ("Pred Energy", net.energy(pred[0]))
            # print(f"Num Trained {train_idx + 1}, Passed:", passed)

            if not passed:
                return train_idx

    print (tryToTrainPredict(images, 0.2))

    randomImages = sign(np.random.randn(200, N))

    print ("\n", "Random")

    print (tryToTrainPredict(randomImages, 0.2))
