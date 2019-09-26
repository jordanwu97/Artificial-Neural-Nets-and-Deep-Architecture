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

    percents = np.round(np.arange(0.05, 1.01, .05 ),2)

    for i, image_original in enumerate(images):
        for j, noise_percent in enumerate(percents):
            args = np.random.choice(N, int(noise_percent * N), replace=False)
            image_noise = np.copy(image_original)
            image_noise[args] = image_noise[args] * -1

            predict = net.predict_sync(image_noise)
            
            print (f"Noise: {noise_percent} Recovered:", np.all(predict==image_original))

            plt.subplot(4,5,j+1)

            showImage(predict, title=f"Noise={noise_percent}", show_image=False)
        
        plt.suptitle(f"P{i+1}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"pictures/3_4_P{i+1}_noise.png")