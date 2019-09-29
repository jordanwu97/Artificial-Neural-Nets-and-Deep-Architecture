import matplotlib.pyplot as plt
import numpy as np
from _3_1 import Hopfield

def loadImgs():
    with open("pict.dat", "r") as f:
        dat = f.read().split(",")   
    return np.reshape(dat, (len(dat)//1024,1024)).astype(int)

def showImage(data_array, title=None, save_file=None, show_image=True, wait=None):
    plt.axis("off")
    plt.imshow(np.reshape(data_array,(32,32)))
    if title:
        plt.title(title)
    if save_file:
        plt.savefig(save_file)
    if show_image:
        plt.show(block=(wait==None))
        if wait:
            plt.pause(wait)
            plt.close()

if __name__ == "__main__":

    images = loadImgs()

#    for dat in images[[1,2,10]]:
#       showImage(dat)

    # train p1 p2 p3
    net = Hopfield()
    net.train(images[:3])

    # Check patterns are stable
    print ("trained stable:", np.all(net.predict_sync(images[:3]) == images[:3]))

    # Check p10 works
    p = net.predict_sync(images[9])
    print ("p10 degraded p1 recovered:", np.all(p == images[0]))
    showImage(p, save_file="pictures/3_2_p10_converge.png", show_image=False)


    # Check p11 works
    p = net.predict_sync(images[10], max_iter=5000)
    print ("p11 recovers p2:", np.all(p == images[1]))
    print ("p11 recovers p3:", np.all(p == images[2]))
    showImage(p, save_file="pictures/3_2_p11_converge.png", show_image=False)

    #p = np.copy(images[9])
    p = np.random.randint(2,size=1024) *2 -1

    for i in range(10):
        p = net.predict_async(p, max_iter=500)
        plt.subplot(2,5,i+1)
        showImage(p, title=f"{(i+1)*500}", show_image=False)

    plt.suptitle(f"P9")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"pictures/3_2_p_rand_iterations.png")
    