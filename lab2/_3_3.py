from _3_1 import *
import matplotlib.markers as markers

from sklearn.metrics.pairwise import rbf_kernel

X = np.arange(0, 2*np.pi, 0.1)
Y_SIN = np.sin(2 * X)
shape = 11

rbfs_mean = np.random.sample(shape) *2*np.pi
rbfs_variance = np.zeros(shape) + 3
rbfs_winner = np.zeros(shape)

eta = 0.1

def winner_takes_all():
    global rbfs_mean
    global rbfs_variance
    global rbfs_winner
    global eta

    n = RBF_NET(rbfs_mean, rbfs_variance)
    n.W = np.random.sample(shape) -0.5

    print("starting pos ", np.sort(rbfs_mean))

    update_loss = []



    for _ in range(20):
        epoch_loss = 0
        for sample_index in range(X.shape[0]):
            min_index = np.argmin(np.absolute(rbfs_mean - X[sample_index]))

            #update
            delta_mean =  -eta*(rbfs_mean[min_index] - X[sample_index])
            epoch_loss += np.absolute(delta_mean)
            rbfs_mean[min_index] += delta_mean
            rbfs_winner[min_index] += 1
        update_loss.append(epoch_loss)

    s_rbfs_mean = np.sort(rbfs_mean)
    print("end pos ", s_rbfs_mean)
    print("num of winning ", rbfs_winner)
    min_dist = 10
    max_dist = 0
    mean_dist = 0
    for i in range(s_rbfs_mean.shape[0]-1):
        dist = s_rbfs_mean[i+1] - s_rbfs_mean[i]
        if(max_dist < dist):
            max_dist = dist
        if(min_dist > dist):
            min_dist = dist
        mean_dist += dist
    mean_dist = mean_dist/(s_rbfs_mean.shape[0]-1)
    print("max_dist ", max_dist)
    print("min_dist ", min_dist)
    print("mean_dist ", mean_dist)

    plt.plot(update_loss)
    plt.show(block=False)
    plt.title("update_loss")
    plt.show()
    plt.clf()

def shared_winner():
    global rbfs_mean
    global rbfs_variance
    global eta

    n = RBF_NET(rbfs_mean, rbfs_variance)
    n.W = np.random.sample(shape) -0.5

    print("starting pos ", np.sort(rbfs_mean))
    #print("starting pos ", rbfs_mean)

    update_loss = []

    for _ in range(20):
        epoch_loss = 0
        for sample_index in range(X.shape[0]):
            distance = (np.absolute(rbfs_mean - X[sample_index]))**2
            #print(X[sample_index])
            #print(max_dist)
            #print(distance)
            min_dist= np.min(distance)

            #update
            delta_mean =  -eta*min_dist/distance*(rbfs_mean - X[sample_index])
            epoch_loss += np.average(np.absolute(delta_mean))
            rbfs_mean += delta_mean
        update_loss.append(epoch_loss)

    #calculate distribution
    s_rbfs_mean = np.sort(rbfs_mean)
    print("end pos ", s_rbfs_mean)
    min_dist = 10
    max_dist = 0
    mean_dist = 0
    for i in range(s_rbfs_mean.shape[0]-1):
        dist = s_rbfs_mean[i+1] - s_rbfs_mean[i]
        if(max_dist < dist):
            max_dist = dist
        if(min_dist > dist):
            min_dist = dist
        mean_dist += dist
    mean_dist = mean_dist/(s_rbfs_mean.shape[0]-1)
    print("max_dist ", max_dist)
    print("min_dist ", min_dist)
    print("mean_dist ", mean_dist)

    plt.plot(update_loss)
    plt.show(block=False)
    plt.title("update_loss")
    plt.show()
    plt.clf()

def ballistic():
    #initialisation
    f = open("data/ballist.dat")
    d = f.read().split("\n")
    d = [[k.split() for k in r.split("\t")] for r in d[:-1]]
    d = np.array(d, dtype=float).reshape(-1, 4)

    X, Y = np.split(d, 2, 1)

    f = open("data/balltest.dat")
    d = f.read().split("\n")
    d = [[k.split() for k in r.split("\t")] for r in d[:-1]]
    d = np.array(d, dtype=float).reshape(-1, 4)

    X_test, Y_test = np.split(d, 2, 1)

    num_rbfs = 9 #only use quadratnumbers
    rbfs_mean = np.zeros((num_rbfs,2))
    rbfs_shape = np.sqrt(num_rbfs)

    for i in range(num_rbfs):
        rbfs_mean[i] = [1/rbfs_shape*(i%rbfs_shape),1/rbfs_shape*np.trunc(i/rbfs_shape)]
    rbfs_variance = 1


    #CL for rbf distribution
    update_loss = []


    for _ in range(20):
        epoch_loss = 0
        for index in range(X.shape[0]):
            dist = (rbfs_mean[:,0] - X[index,0])**2 + (rbfs_mean[:,1] - X[index,1])**2
            dist = np.square(dist)
            min_index = np.argmin(dist)

            # update
            delta_mean = -eta * (rbfs_mean[min_index,:] - X[index,:])
            epoch_loss += dist[min_index]
            rbfs_mean[min_index,:] += delta_mean
            #rbfs_winner[min_index] += 1
        update_loss.append(epoch_loss)


    plot = True
    if (plot):
        plt.plot(X[:,0], X[:,1], "o", color="orange")
        plt.plot(rbfs_mean[:, 0], rbfs_mean[:, 1], "x", color="blue")
        #plt.plot(Y[:,0], Y[:,1], "s")
        plt.show(block=False)
        plt.title("rbf_distribution")
        plt.show()
        plt.clf()


    #train weights
    n = RBF_NET(rbfs_mean, rbfs_variance)

    n.train_batch(X,Y)
    print(n.mae(X_test, Y_test))

    vis = np.arange(0,1,0.01)
    X_vis = np.zeros((vis.shape[0],2))
    X_vis[:, 0] = vis[:]
    X_vis[:, 1] = 0.8
    Y_vis= n.predict(X_vis)
    furtherst = np.argmax(Y[:,0])


    Y_test_pred= n.predict(X_test)


    plot = True
    if (plot):
        plt.plot(Y_vis[:,0], Y_vis[:,1], "o")
        plt.xlabel("distance")
        plt.ylabel("height")
        #plt.plot(Y_test_pred[:20,0], Y_test_pred[:20,1], "o", label="prediction")
        #plt.plot(Y_test[:20,0], Y_test[:20,1], ".", label="actual data")
        plt.legend()
        plt.show(block=False)
        #plt.title("Prediction for test data")
        plt.title("Prediction for different angles")
        plt.show()
        plt.clf()

def plot_endvalues():
    pos_1 = [0.46919699, 1.24175863, 1.94175953, 2.52098496, 2.96108921, 3.38483988, 3.87803621, 4.40677224, 4.98049918, 5.58052388, 6.06320906]
    pos_2 = [0.55873653, 1.14425698, 1.67399974, 2.18035987, 2.67162994, 3.15776469, 3.64967936, 4.1527186, 4.67934174, 5.23261839, 5.83893725]

    plt.plot(pos_1, np.zeros(11), "|", color="green", label="one_winning")
    plt.plot(pos_2, np.zeros(11)+0.1, "|", color="red", label="shared_winning")
    plt.plot(pos_1, np.zeros(11)-0.005, "-", color="green")
    plt.plot(pos_2, np.zeros(11)+0.105, "-", color="red")
    plt.plot(7, 0.5, ".", color="white") #for alibi
    plt.axis(xlim=(-0.1, 7), ylim=(0, 1))
    plt.show(block=False)
    plt.title("rbf_distribution")
    plt.legend()
    plt.show()
    plt.clf()

if __name__ == "__main__":
    ballistic()

