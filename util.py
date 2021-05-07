import numpy as np
import numpy.linalg as linalg
from progress.bar import Bar
import matplotlib.pyplot as plt
from parameters import *


"""
distance(posA, posB): distance between particle A and particle B
msd(posData): mean square displacement (dynamic property)
vsd(posData, msd): variance square displacement (dynamic property)
mnn_distance(posData): mean nearest neighbour distance (static property)
vnn_distance(posData, mnn_distance): variance nearest neighbour distance (static property)
calc_rdf(pos, pType): radial distribution function for a single timestep
"""


def calc_distance(posA, posB):
    """
    calculates the distances between two particles taking the periodic boundary
    into consideration
    """

    L = 28.9

    # calculate the distance vector
    r1 = abs(posA - posB)

    # periodic boundary condition
    r2 = abs(r1 - L)
    r = np.amin([r1, r2], axis=0) 

    r = linalg.norm(r)
    return r


def msd(posData):
    """
    calculates the mean square displacement for each timestep
    """

    posZero = posData[0]
    f = lambda x: linalg.norm(x - posZero, axis=2)**2
    msd = np.mean(f(posData), axis=1)

    print('Mean square displacement calculated...')
    return msd


def vsd(posData, msd):
    """
    calculates the variance square displacement for each timestep
    """

    posZero = posData[0]
    f = lambda x: linalg.norm(x-posZero, axis=2)**4
    variance = np.mean(f(posData), axis=1) - msd**2

    print('Variance square displacement calculated...')
    return variance


def mean_nn(posData, cutoff):
    """
    calculates the mean nearest neighbour distance and the mean amount
    of neighbours within the cutoff
    """
    nn_distance = np.zeros(len(posData))
    nn_amount = np.zeros(len(posData))

    bar = Bar('calc. mnn distance..', max=len(posData))

    for timestep, timestepPos in enumerate(posData):
        nn = [np.inf] * particles
        nn2 = [0] * particles
        
        # calculate the top diagonal of the distance matrix
        for i, pos in enumerate(timestepPos):
            for j, pos2 in enumerate(timestepPos[:i]):

                # calculate the distance between index1 and index2
                distance = calc_distance(pos, pos2)

                if distance < cutoff:
                    nn2[i] += 1
                    nn2[j] += 1

                nn[i] = min(nn[i], distance)
                nn[j] = min(nn[j], distance)
        

        # calculate the mean of the nearest neighbour distance
        nn_distance[timestep] = np.mean(nn)
        nn_amount[timestep] = np.mean(nn2)
        bar.next()
    
    bar.finish()
    return nn_distance, nn_amount

def variance_nn(posData, mnn_distance, mnn_amount, cutoff):
    """
    calculates the variance of the nearest neighbour distance
    """
    

    nn_distance = np.zeros(len(posData))
    nn_amount = np.zeros(len(posData))

    bar = Bar('calc. vnn distance..', max=len(posData))

    for timestep, timestepPos in enumerate(posData):
        nn = [np.inf] * particles
        nn2 = [0] * particles

        
        # calculate the top diagonal of the distance matrix
        for i, pos in enumerate(timestepPos):
            for j, pos2 in enumerate(timestepPos[:i]):
                
                # calculate the squared distance between particles i and j
                distance = np.square(calc_distance(pos, pos2))

                if distance < cutoff:
                    nn2[i] += 1
                    nn2[j] += 1


                nn[i] = min(nn[i], distance)
                nn[j] = min(nn[j], distance)

        nn_distance[timestep] = np.mean(nn) - np.square(mnn_distance[timestep])
        nn_amount[timestep] = np.mean(np.square(nn2)) - np.square(mnn_amount[timestep])

        bar.next()

    bar.finish()
    return nn_distance, nn_amount


def calc_mean(Data):
    """
    calculates the mean of the norm of the vector per timestep
    """
    f = lambda x: linalg.norm(x, axis=2)
    mean = np.mean(f(Data), axis=1)

    print('Mean calculated...')
    return mean


def calc_variance(Data, mean):
    """
    calculates the variance of the norm of the vector per timestep
    """
    f = lambda x: np.square(linalg.norm(x, axis=2))
    variance = np.mean(f(Data), axis=1) - np.square(mean)

    print('Variance calculated...')
    return variance


def calc_rdf(pos, pType):
    """
    calculates the radial distribution function
    @param :pos: position of all the particles in a single timestep
    @param :pType: list of particle types, 1 or 2
    """


    r         = np.arange(0,rmax+dr,dr)
    NR        = len(r)
    grAA      = np.zeros((NR, 1))
    grBB      = np.zeros((NR, 1))
    grAB      = np.zeros((NR, 1))

    iA = np.where(pType==1)[0]
    iB = np.where(pType==2)[0]
    NA = len(iA)
    NB = len(iB)

    for i in range(particles):
        for j in range(i+1, particles):
            
            rx = abs(pos[i, 0] - pos[j, 0])
            ry = abs(pos[i, 1] - pos[j, 1])
                 
            # periodic boundary conditions --> rij = min( rij, abs(rij-L) )
            if rx > 0.5*Lx:
                rx = abs(rx - Lx)
            if ry > 0.5*Ly:
                ry = abs(ry - Ly)

            r = np.sqrt(rx**2 + ry**2)

            if r <= rmax:
                igr = round(r/dr)

                if i in iA and j in iA:
                    grAA[igr] = grAA[igr] + 2
                elif i in iB and j in iB:
                    grBB[igr] = grBB[igr] + 2
                else:
                    grAB[igr] = grAB[igr] + 1

    # normalize
    dr2       = np.zeros((NR,1))

    for ir in range(NR):
        rlow    = ir*dr
        rup     = rlow + dr
        dr2[ir] = rup**2 - rlow**2

    nidealA   = np.pi * dr2 * (NA*NA/A)            # g(r) for ideal gas of A particles
    nidealB   = np.pi * dr2 * (NB*NB/A)            # g(r) for ideal gas of B particles
    nidealAB  = np.pi * dr2 * (NA*NB/A)            # g(r) for ideal gas of A+B particles

    grAA_norm = grAA / nidealA
    grBB_norm = grBB / nidealB
    grAB_norm = grAB / nidealAB
    r         = np.arange(0,rmax+dr,dr)


    return grAA_norm, grBB_norm, grAB_norm

def calc_rdf_peaks(posData, types):
    """
    calculates the radial distribution function peak for AA, AB and BB.
    Only for AA and AB does this value also correspond to the first peak
    """

    grAA_amax, grBB_amax, grAB_amax = np.zeros(len(posData))

    bar = Bar('calc. rdf peaks..', max=len(posData))

    for t, pos_t, type_t in zip(len(posData), posData, types):

        grAA, grBB, grAB = calc_rdf(pos_t, type_t)

        # calculate the argmax corresponding to the first peak for grAA and grAB
        grAA_amax[t] = np.argmax(grAA)
        grBB_amax[t] = np.argmax(grBB)
        grAB_amax[t] = np.argmax(grAB)

        bar.next()

    bar.finish()
    
    return grAA_amax, grBB_amax, grAB_amax


def calc_avg_rdf(posData, types):
    """
    calculates the average rdf over all the timesteps
    """

    r         = np.arange(0,rmax+dr,dr)
    NR        = len(r)
    grAA      = np.zeros((NR, 1))
    grBB      = np.zeros((NR, 1))
    grAB      = np.zeros((NR, 1))

    bar = Bar('calc. rdf peaks..', max=len(posData))

    for pos_t, type_t in zip(posData, types):

        grAAt, grBBt, grABt = calc_rdf(pos_t, type_t)
        grAA += grAAt
        grBB += grBBt
        grAB += grABt

        bar.next()

    bar.finish()

    grAA /= len(posData)
    grBB /= len(posData)
    grAB /= len(posData)

    return grAA, grBB, grAB


def plot_rdf(r, gr, title):
    """
    plots a radial distribution function
    """
    plt.plot(r, gr)
    plt.title(title)
    plt.xlabel('r')
    plt.ylabel('gr')
    plt.show()


def save_load(func, savename):
    """
    tries to load the given funcion from npy files, if it is not there
    run the function
    """
    try:
        with open(savename, 'rb') as f:
            result = np.load(f)
            print('Feature loaded..')
    except:
        result = np.asarray(func())
        with open(savename, 'wb') as f:
            np.save(f, result)

    return result
