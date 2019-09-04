import matplotlib.pyplot as plt

def plotLearningCurve(losses):
    """
    Given a list of losses at each epoch, plot the learning curve
    """
    plt.plot([i+1 for i in range(len(losses))], losses)
    plt.title("Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.show()