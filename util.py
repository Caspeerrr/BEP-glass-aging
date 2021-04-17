import numpy as np
import numpy.linalg as linalg
from progress.bar import Bar


"""
msd(posData): mean square displacement (dynamic property)
vsd(posData, msd): variance square displacement (dynamic property)
mnn_distance(posData): mean nearest neighbour distance (static property)
vnn_distance(posData, mnn_distance): variance nearest neighbour distance (static property)
"""

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
    N = len(posData[0])

    bar = Bar('calc. mnn distance..', max=len(posData))

    for timestep, timestepPos in enumerate(posData):
        nn = [np.inf] * N
        nn2 = [0] * N
        
        # calculate the top diagonal of the distance matrix
        for i, pos in enumerate(timestepPos):
            for j, pos2 in enumerate(timestepPos[:i]):

                # calculate the distance between index1 and index2
                distance = linalg.norm(pos - pos2)

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
    N = len(posData[0])

    bar = Bar('calc. vnn distance..', max=len(posData))

    for timestep, timestepPos in enumerate(posData):
        nn = [np.inf] * N
        nn2 = [0] * N

        
        # calculate the top diagonal of the distance matrix
        for i, pos in enumerate(timestepPos):
            for j, pos2 in enumerate(timestepPos[:i]):
                
                # calculate the squared distance between particles i and j
                distance = np.square(linalg.norm(pos - pos2))

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
