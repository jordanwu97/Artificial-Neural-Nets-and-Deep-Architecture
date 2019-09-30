from _3_1 import Hopfield, sign
from _3_2 import loadImgs, showImage
from _3_3 import plotCurves
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    images = loadImgs()[:9]
    images = np.flip(images, axis=0)
    N = images.shape[1]

    def predictStableSamples(_images, noiseLevel):
        performance = []

        stable_images = []

        for train_idx in range(len(_images)):

            target = _images[:train_idx + 1]

            net = Hopfield()
            net.train(target)


            #last attempt still stable?
            sum_stable = 0
            if train_idx > 0:
                pred = net.predict_sync(stable_images, max_iter=1)
                for index in range(train_idx):
                    if (net.energy(pred[index]) == net.energy(stable_images[index])):
                        sum_stable += 1
                sum_stable = sum_stable/(train_idx)
                performance.append(sum_stable)

            # calculate new stable images
            # add noise
            args = np.random.choice(N, int(noiseLevel * N), replace=False)
            noisy = np.copy(target)
            noisy[:, args] = noisy[:, args] * -1

            stable_images = net.predict_sync(noisy)
            passed = np.all(stable_images == target)

            print(train_idx, "sum_stable", sum_stable)#, "passed: ", passed)


        return performance


    def tryToTrainPredict(_images, noiseLevel, delete_diagonal=False):

        performance = []

        for train_idx in range(len(_images)):
            print(train_idx)
            target = _images[:train_idx + 1]

            net = Hopfield()
            net.train(target, delete_diagonal)

            # try reconverging the last image (target) from a noisy version of target
            args = np.random.choice(N, int(noiseLevel * N), replace=False)
            noisy = np.copy(target)
            noisy[:,args] = noisy[:,args] * -1
            pred = net.predict_sync(noisy)

            passed = np.all(pred == target)

            # performance rate
            sum = 0
            for e in range(train_idx+1):
                if(net.energy(target[e]) == net.energy(pred[e])):
                    sum += 1

            sum = sum/(train_idx+1)
            performance.append(sum)


            # print ("Target Energy", net.energy(target[0]))
            # print ("Pred Energy", net.energy(pred[0]))
            # print(f"Num Trained {train_idx + 1}, Passed:", passed)

            #if not passed:
                #return train_idx
        return performance


#    plotCurves(
#        {"Performance": tryToTrainPredict(images, 0.2)},
#        "Images trained",
#        "identified correct",
#        "Performance per trained Images (given samples)",
#        save_file="pictures/3_5_performance_given_images.png",
#    )

    randomImages = sign(np.random.randn(200, N))

#    print ("\n", "Random")
#    plotCurves(
#            {"Performance": tryToTrainPredict(randomImages, 0.2)},
#            "Images trained",
#            "identified correct",
#            "Performance per trained Images (random samples)",
#            save_file="pictures/3_5_performance_random_images.png",
#        )

    randomImages = sign(0.5+np.random.randn(100, N))

    print(randomImages.shape)
    print("\n", "Removed self connections")
    plotCurves(
        {"Performance": tryToTrainPredict(randomImages, 0.2, True)},
        "Images trained",
        "identified correct",
        "Performance per trained Images (random samples)",
        save_file="pictures/3_5_performance_wo_wii_images_bias.png",
    )

    N = 100
    randomImages = sign(np.random.randn(300, N))
    print(randomImages.shape)

    print ("\n", "stable after adding to Weight")
   # plotCurves(
   #         {"Performance": predictStableSamples(randomImages, 0.2)},
   #         "Images trained",
   #         "stable samples",
   #         "Performance per trained Images (random samples)",
   #         save_file="pictures/3_5_performance_noise_stable_samples.png",
   #     )

