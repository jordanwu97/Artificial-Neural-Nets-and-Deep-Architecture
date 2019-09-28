from _3_1 import Hopfield, sign
from _3_2 import loadImgs, showImage
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    images = loadImgs()[:9]
    images = np.flip(images, axis=0)
    N = images.shape[1]

    for train_idx in range(len(images)):
        to_train = images[:train_idx + 1]

        net = Hopfield()
        net.train(to_train)

        # try reconvering the last image (target) from a noisy version of target
        target = to_train[0]

        args = np.random.choice(N, int(0 * N), replace=False)
        noisy = np.copy(target)
        noisy[args] = noisy[args] * -1

        pred = net.predict_sync(noisy)

        # [ showImage(p) for p in pred ]

        # print ("Target Energy", net.energy(target))
        # print ("Pred Energy", net.energy(pred))

        # rememberedAll = [ pred == target ] 

        print(f"Num Trained {train_idx + 1}, Passed:", np.all(pred == target))
