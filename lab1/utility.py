import matplotlib.pyplot as plt

def plotLearningCurve(*losses):
    """
    Given a list of losses at each epoch, plot the learning curve
    """
    for label, loss in losses:
        plt.plot([i+1 for i in range(len(loss))], loss, label=label)
    plt.title("Learning Curve")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("MSE(Log)")
    plt.legend()
    plt.show()

def plotLearningCurves(*losses):
    """
    Given a list of losses at each epoch, plot the learning curve
    """
    label = " "
    for value in losses:
        for i in range(len(value)):
            if(i==0):
                label = value[i]
            else:
                plt.plot([e+1 for e in range(len(value[i]))], value[i], label=label[i-1])
    plt.title("Learning Curve")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("MSE(Log)")
    plt.legend()
    plt.show()

