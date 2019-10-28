from _4_1 import *
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm

def loadVoting():
    with open("data/mpnames.txt", encoding = "ISO-8859-1") as f:
        mp_name = np.array(f.read().split("\n")[:-1])

    with open("data/mpdistrict.dat") as f:
        mp_dist = np.array(f.read().split("\n")[:-1])
        
    with open("data/mpparty.dat") as f:
        mp_party = np.array(f.read().split("\n")[3:-1])

    with open("data/mpsex.dat") as f:
        mp_sex = np.array(f.read().split("\n")[2:-1])

    with open("data/votes.dat") as f:
        vote = f.read().split(",")
        vote = np.array(vote, dtype=float)
        vote = vote.reshape(len(mp_name), -1)
    
    return mp_name, mp_dist, mp_party, mp_sex, vote

if __name__ == "__main__":

    # value = -1
    # data = np.arange(100).reshape((10, 10))
    # data[5, :] = -1  # Values to set -1

    # masked_array = np.ma.masked_where(data == value, data)

    # print (masked_array)
    # exit()

    names, dist, party, sex, votes = loadVoting()

    nodes_shape = 10, 10
    num_epochs = 50

    def neighbor_size(ep):
        if ep < (num_epochs * 1/3):
            return 4
        if ep < (num_epochs * 2/3):
            return 2
        return 1

    neighborhoodmap = np.array([[[a,b] for b in range(10)] for a in range(10) ]).reshape(100,2)

    def neigh_func(center, neighborhood):
        center_v = neighborhoodmap[center]
        # dist_mat = np.linalg.norm(neighborhoodmap - center_v, axis=2)
        # args = np.where(dist_mat <= neighborhood)
        # print (center)
        out = rbf_kernel(neighborhoodmap, center_v, gamma=1/((neighborhood)**2))

        return out

    som = SOM(nodes_shape[0] * nodes_shape[1], neigh_func, neighbor_size)
    som.train(votes, num_epochs)
    ok = som.showMap(votes, party)

    # print (som.map)

    partymapping = [ [] for _ in range(100) ]

    for p,idx in zip(party, som.map):
        partymapping[idx].append(p)
    partymapping = np.array([ (list(stats.mode(a)[0]) + [-1])[0] for a in partymapping ], dtype=int).reshape(10,10)

    partymapping = np.ma.masked_where(partymapping == -1, partymapping)
    # print (partymapping)
    plt.imshow(partymapping, cmap="Set1")
    plt.colorbar()
    plt.savefig("pictures/partymapping.png")
    plt.clf()

    
    districtmapping = [ [] for _ in range(100) ]
    for p,idx in zip(dist, som.map):
        districtmapping[idx].append(p)
    districtmapping = np.array([ (list(stats.mode(a)[0]) + [-1])[0] for a in districtmapping ], dtype=int).reshape(10,10)
    districtmapping = np.ma.masked_where(districtmapping == -1, districtmapping)
    # print (districtmapping)

    plt.imshow(districtmapping, cmap="tab20")
    plt.colorbar()
    plt.savefig("pictures/districtmapping.png")
    plt.clf()

    sexmapping = [ [] for _ in range(100) ]
    for p,idx in zip(sex, som.map):
        sexmapping[idx].append(p)

    sexmapping = np.array([ np.mean(np.array(a, dtype=int)) for a in sexmapping ]).reshape(10,10)
    
    # exit()

    # sexmapping = np.array([ (list(stats.mode(a)[0]) + [-1])[0] for a in sexmapping ], dtype=int).reshape(10,10)
    sexmapping = np.ma.masked_invalid(sexmapping)
    print (sexmapping)

    plt.imshow(sexmapping, cmap="cool")
    plt.colorbar()
    plt.savefig("pictures/sexmapping.png")

