import matplotlib.pyplot as plt
import numpy as np
from _3_1 import Hopfield

def loadImgs():
    with open("pict.dat", "r") as f:
        dat = f.read().split(",")   
    return np.reshape(dat, (len(dat)//1024,1024)).astype(int)

def showImage(data_array, save_file=None, show_image=True):
    plt.clf()
    plt.imshow(np.reshape(data_array,(32,32)))
    if save_file:
        plt.savefig(save_file)
    if show_image:
        plt.show()

if __name__ == "__main__":

    images = loadImgs()

    # for dat in images[[1,2,10]]:
    #     showImage(dat)

    # train p1 p2 p3
    net = Hopfield()
    net.train(images[:3])

    # # Check patterns are stable
    # print ("trained stable:", np.all(net.predict_sync(images[:3]) == images[:3]))

    # # Check p10 works
    # p = net.predict_sync(images[9])
    # print ("p10 degraded p1 recovered:", np.all(p == images[0]))
    # showImage(images[0])
    # showImage(p, "pcitures/3_2_p10_converge.png")


    # # Check p11 works
    # p = net.predict_sync(images[10], max_iter=5000)
    # print ("p11 recovers p2:", np.all(p == images[1]))
    # print ("p11 recovers p3:", np.all(p == images[2]))
    # showImage(p, "pictures/3_2_p11_converge.png")

    p = np.copy(images[9])
    for i in range(10):
        p = net.predict_async(p, max_iter=500)
        showImage(p, save_file=f"pictures/3_2_async_image10_iter={i*500}.png", show_image=False)