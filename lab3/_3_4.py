from _3_1 import Hopfield, sign
from _3_2 import loadImgs, showImage
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    images = loadImgs()[:3]

    N = images.shape[1]

    # train p1 p2 p3
    net = Hopfield()
    net.train(images)

    percents = np.round(np.arange(0.05, 1.01, 0.05), 2)

    print("Image & Max Noise Mean & Max Noise Std")

    # For different images check
    for i, image_original in enumerate(images):

        max_noise = []

        for k in range(50):

            max_noise_recoverable = 0

            for j, noise_percent in enumerate(percents):

                args = np.random.choice(N, int(noise_percent * N), replace=False)
                image_noise = np.copy(image_original)
                image_noise[args] = image_noise[args] * -1

                predict = net.predict_sync(image_noise)
                recovered = np.all(predict == image_original)
                max_noise_recoverable = (
                    noise_percent if recovered else max_noise_recoverable
                )

                if k == 0:
                    plt.subplot(4, 5, j + 1)
                    showImage(predict, title=f"Noise={noise_percent}", show_image=False)

            max_noise.append(max_noise_recoverable)

        print(f"P{i+1} & {np.mean(max_noise)} & {np.std(max_noise)}")
        plt.suptitle(f"P{i+1}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"pictures/3_4_P{i+1}_noise.png", bbox_inches='tight')
        plt.clf()
