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