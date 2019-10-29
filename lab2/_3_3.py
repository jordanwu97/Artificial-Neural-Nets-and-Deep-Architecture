from _3_1 import *

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

    #print("starting pos ", np.sort(rbfs_mean))
    print("starting pos ", rbfs_mean)

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
        plt.plot(rbfs_mean[:,0], rbfs_mean[:,1], "x")
        plt.plot(X[:,0], X[:,1], "o")
        #plt.plot(Y[:,0], Y[:,1], "s")
        plt.show(block=False)
        plt.title("rbf_distribution")
        plt.show()
        plt.clf()


    #train weights
    n = RBF_NET(rbfs_mean, rbfs_variance)

    n.train_batch(X,Y)
    print(n.mae(X_test, Y_test))

    # vis = np.arange(0,1,0.01)
    # X_vis = np.zeros((vis.shape[0],2))
    # X_vis[:,0] = vis[:]
    # X_vis[:,1] = 0.45
    # # for i in range(vis.shape[0]):
    # #     X_vis[i,0] = vis[i]
    # #     X_vis[i,1] = 0.45


    Y_test_pred= n.predict(X_test)


    plot = True
    if (plot):
 #       plt.plot(X_vis[:,0], X_vis[:,1], "o")
        plt.plot(Y_test_pred[10:20,0], Y_test_pred[10:20,1], "o")
        plt.plot(Y_test[10:20,0], Y_test[10:20,1], ".")
        plt.show(block=False)
        plt.title("Prediction for different angels")
        plt.show()
        plt.clf()


if __name__ == "__main__":
    ballistic()






#Y = np.arange(20)
#X = np.arange(10)

#var = 3

#n = RBF_NET(Y, 3, sign)

#phi = n.phi(X)

#phi2 = rbf_kernel(X.reshape(-1,1), Y.reshape(-1,1), gamma=1/(2*var**2))
#print (np.allclose(phi2,phi))
