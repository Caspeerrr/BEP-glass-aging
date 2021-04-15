import numpy as np
import numpy.linalg as linalg

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
    f = lambda x: np.square(linalg.norm(x - posZero, axis=2))
    msd = np.mean(f(posData), axis=1)

    print('Mean square displacement calculated...')
    return msd


def vsd(posData, msd):
    """
    calculates the variance square displacement for each timestep
    """

    posZero = posData[0]
    f = lambda x: linalg.norm(x-posZero, axis=2)** 4
    variance = np.mean(f(posData), axis=1) - msd**2

    print('Variance calculated...')
    return variance


def mnn_distance(posData):
    """
    calculates the mean nearest neighbour distance
    """

    nn_distance = np.zeros(len(posData))
    N = len(posData[0])

    for timestep, timestepPos in enumerate(posData):
        
        nn = [np.inf] * N
        
        # calculate the top diagonal of the distance matrix
        for index, pos in enumerate(timestepPos):
            for index2, pos2 in enumerate(timestepPos[:index]):

                # calculate the distance between index1 and index2
                distance = linalg.norm(pos - pos2)

                nn[index] = min(nn[index], distance)
                nn[index2] = min(nn[index2], distance)

        # calculate the mean of the nearest neighbour distance
        nn_distance[timestep] = np.mean(nn)

    print('Mean nearest neighbor distance calculated...')
    return nn_distance


def vnn_distance(posData, mnn_distance):
    """
    calculates the variance of the nearest neighbour distance
    """

    nn_distance = np.zeros(len(posData))
    N = len(posData[0])

    for timestep, timestepPos in enumerate(posData):
        
        nn = [np.inf] * N
        
        # calculate the top diagonal of the distance matrix
        for i, pos in enumerate(timestepPos):
            for j, pos2 in enumerate(timestepPos[:i]):
                
                # calculate the squared distance between particles i and j
                distance = np.square(linalg.norm(pos - pos2))

                nn[i] = min(nn[i], distance)
                nn[j] = min(nn[j], distance)

        nn_distance[timestep] = np.mean(nn) - np.square(mnn_distance[timestep])
    
    print('Variance nearest neighbor distance calculated...')
    return nn_distance


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
