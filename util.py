import numpy as np
import numpy.linalg as linalg
from progress.bar import Bar
import matplotlib.pyplot as plt
from init import *

"""
distance(posA, posB): distance between particle A and particle B
msd(posData): mean square displacement (dynamic property)
vsd(posData, msd): variance square displacement (dynamic property)
mnn_distance(posData): mean nearest neighbour distance (static property)
vnn_distance(posData, mnn_distance): variance nearest neighbour distance (static property)
calc_rdf(pos, pType): radial distribution function for a single timestep
"""


# def calc_distance(posA, posB):
#     """
#     calculates the distances between two particles taking the periodic boundary
#     into consideration
#     """

#     L = 28.9

#     # calculate the distance vector
#     r1 = abs(posA - posB)

#     # periodic boundary condition
#     r2 = abs(r1 - L)
#     r = np.amin([r1, r2], axis=0) 

#     r = linalg.norm(r)
#     return r

def calc_distance(posA, posB):
    """
    calculates the distance between two particles taking the periodic boundary
    into consideration
    """

    L = 28.9

    rx = abs(posA[0] - posB[0])
    ry = abs(posA[1] - posB[1])

    # periodic boundary condition
    if rx > 0.5*L:
        rx = abs(rx - L)
    if ry > 0.5*L:
        ry = abs(ry - L)
    
    r = np.sqrt(rx**2 + ry**2)

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
        nn = [np.inf] * params['particles']
        nn2 = [0] * params['particles']
        
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

def mean_nn2(posData, cutoff):
    """
    calculates the mean nearest neighbour distance and the mean amount
    of neighbours within the cutoff
    """
    nn_distance = np.zeros(len(posData))
    nn_amount = np.zeros(len(posData))

    bar = Bar('calc. mnn distance..', max=len(posData))

    for timestep, timestepPos in enumerate(posData):
        nn = [np.inf] * params['particles']
        nn2 = [0] * params['particles']
        

        for i in range(params['particles']):
            for j in range(i+1, params['particles']):

                distance = calc_distance(timestepPos[i], timestepPos[j])
            
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
        nn = [np.inf] * params['particles']
        nn2 = [0] * params['particles']

        
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

    dr        = params['dr']
    rmax      = params['rmax']
    A         = params['A']
    Lx        = params['Lx']
    Ly        = params['Ly']
    r         = np.arange(0,rmax+dr,dr)
    NR        = len(r)
    grAA      = np.zeros((NR, 1))
    grBB      = np.zeros((NR, 1))
    grAB      = np.zeros((NR, 1))

    iA = np.where(pType==1)[0]
    iB = np.where(pType==2)[0]
    NA = len(iA)
    NB = len(iB)

    for i in range(params['particles']):
        for j in range(i+1, params['particles']):
            
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

def calc_rdf_all(posData, types):
    """
    calculates the radial distribution function peak for AA, AB and BB for
    all timesteps
    """

    grAA, grBB, grAB = np.zeros(len(posData)), np.zeros(len(posData)), np.zeros(len(posData))

    bar = Bar('calc. rdf peaks..', max=len(posData))

    for t, pos_t, type_t in zip(range(len(posData)), posData, types):

        grAA_t, grBB_t, grAB_t = calc_rdf(pos_t, type_t)

        # calculate the argmax corresponding to the first peak for grAA and grAB
        grAA_amax[t] = np.argmax(grAA)
        grBB_amax[t] = np.argmax(grBB)
        grAB_amax[t] = np.argmax(grAB)

        bar.next()

    bar.finish()
    
    return grAA_amax, grBB_amax, grAB_amax

def calc_rdf_peaks(posData, types):
    """
    calculates the radial distribution function peak for AA, AB and BB.
    Only for AA and AB does this value also correspond to the first peak
    """

    grAA_amax, grBB_amax, grAB_amax = np.zeros(len(posData)), np.zeros(len(posData)), np.zeros(len(posData))

    bar = Bar('calc. rdf peaks..', max=len(posData))

    for t, pos_t, type_t in zip(range(len(posData)), posData, types):

        grAA, grBB, grAB = calc_rdf(pos_t, type_t)

        # calculate the argmax corresponding to the first peak for grAA and grAB
        grAA_amax[t] = np.argmax(grAA)
        grBB_amax[t] = np.argmax(grBB)
        grAB_amax[t] = np.argmax(grAB)

        bar.next()

    bar.finish()
    
    return grAA_amax, grBB_amax, grAB_amax

def calc_rdf_minimum(posData, types):
    """
    calculates the radial distribution function minimum for AA, AB and BB.
    """

    grAA_amin, grBB_amin, grAB_amin = np.zeros(len(posData)), np.zeros(len(posData)), np.zeros(len(posData))

    bar = Bar('calc. rdf minimum..', max=len(posData))

    for t, pos_t, type_t in zip(range(len(posData)), posData, types):

        grAA, grBB, grAB = calc_rdf(pos_t, type_t)

        # calculate the argmax corresponding to the first peak for grAA and grAB
        grAA_amax = np.argmax(grAA)
        grBB_amax = np.argmax(grBB)
        grAB_amax = np.argmax(grAB)

        # reload the arrays but from the maximum index and further
        grAA = grAA[grAA_amax:]
        grBB = grBB[grBB_amax:]
        grAB = grBB[grAB_amax:]

        # calculate the argmin corresponding to the first minimum for grAA, grAB, grBB
        grAA_amin[t] = np.argmin(grAA) + grAA_amax
        grBB_amin[t] = np.argmin(grBB) + grBB_amax
        grAB_amin[t] = np.argmin(grAB) + grAB_amax

        bar.next()

    bar.finish()
    
    return grAA_amin, grBB_amin, grAB_amin


def calc_rdf_area(posData, types):
    """
    calculates the area under the radial distribution function for AA, AB and BB.
    """

    dr = params['dr']
    grAA_area, grBB_area, grAB_area = np.zeros(len(posData)), np.zeros(len(posData)), np.zeros(len(posData))

    bar = Bar('calc. rdf area..', max=len(posData))

    for t, pos_t, type_t in zip(range(len(posData)), posData, types):

        grAA, grBB, grAB = calc_rdf(pos_t, type_t)

        # calculate the area under the rdf graph per timestep
        grAA_area[t] = np.sum(grAA * dr)
        grBB_area[t] = np.sum(grBB * dr)
        grAB_area[t] = np.sum(grAB * dr)

        bar.next()

    bar.finish()

    return grAA_area, grBB_area, grAB_area

def calc_avg_rdf(posData, types):
    """
    calculates the average rdf over all the timesteps
    """

    dr        = params['dr']
    rmax      = params['rmax']
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


def calc_voronoi_peaks(timesteps, vor_area, vor_amn):
    """
    Calculates the magnitude and count of the first two peaks of the voronoi area 
    histogram and of the amount of voronoi edges histogram for each timestep. Returns
    these in numpy arrays for all timesteps.
    """

    area_peak1_count, area_peak2_count = np.zeros(len(timesteps)), np.zeros(len(timesteps))
    area_peak1_mag, area_peak2_mag = np.zeros(len(timesteps)), np.zeros(len(timesteps))
    amount_peak1_count, amount_peak2_count = np.zeros(len(timesteps)), np.zeros(len(timesteps))
    amount_peak1_mag, amount_peak2_mag = np.zeros(len(timesteps)), np.zeros(len(timesteps))

    for timestep in timesteps:

        voronoi_area, voronoi_amount = vor_area[timestep], vor_amn[timestep]

        # save the first two peaks attributes of the voronoi area
        values, bins, _ = plt.hist(voronoi_area, bins=20)
        order = np.argsort(values)[::-1]

        area_peak1_count_t, area_peak2_count_t = values[order][:2]
        area_peak1_mag_t, area_peak2_mag_t = [bins[i] + (bins[i+1] - bins[i])/2 for i in order[:2]]

        area_peak1_count[timestep] = area_peak1_count_t
        area_peak2_count[timestep] = area_peak2_count_t
        area_peak1_mag[timestep] = area_peak1_mag_t
        area_peak2_mag[timestep] = area_peak2_mag_t

        # save the first two peaks attributes of the amount of voronoi edges
        values, bins, _ = plt.hist(voronoi_amount, bins=10)
        order = np.argsort(values)[::-1]

        amount_peak1_count_t, amount_peak2_count_t = values[order][:2]
        amount_peak1_mag_t, amount_peak2_mag_t = [bins[i] + (bins[i+1] - bins[i])/2 for i in order[:2]]

        amount_peak1_count[timestep] = amount_peak1_count_t
        amount_peak2_count[timestep] = amount_peak2_count_t
        amount_peak1_mag[timestep] = amount_peak1_mag_t
        amount_peak2_mag[timestep] = amount_peak2_mag_t

    return area_peak1_count, area_peak2_count, area_peak1_mag, area_peak2_mag, amount_peak1_count, amount_peak2_count, amount_peak1_mag, amount_peak2_mag


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


def calc_SF(posData, types):

    # initialize
    thresh = 1e-12
    dk = 2 * np.pi / params['Lx']
    nmax = 100

    kvec, knorm = np.zeros((nmax**2, 2)), np.zeros((nmax**2, 1))

    count = 0

    # calculate k vectors
    for nx in range(0, nmax, 1):
        for ny in range(0, nmax, 1):

            kvec[count, 0] = nx * dk
            kvec[count, 1] = ny * dk
            knorm[count] = np.linalg.norm(kvec[count])
            count += 1

    kvec = np.delete(kvec, (0), axis=0)
    knorm = np.delete(knorm, (0), axis=0)

    Nk = kvec.shape[0]

    u = np.unique(knorm)
    ind = [0]
    ind2 = np.nonzero(abs(np.diff(u)) >= thresh)[0]
    ind = np.append(ind,ind2+1)
    uk = u[np.ix_(ind)]                   # [u[i] for i in ind] 

    Nuk = len(uk)

    # initialize S(k)
    SkAA = np.zeros(Nk)
    SkBB = np.zeros(Nk)
    skAB = np.zeros(Nk)

    for i in range(Nk):
        
        kx, ky = kvec[i, 0], kvec[i, 1]

        costermA = 0
        sintermA = 0
        costermB = 0
        sintermB = 0

        for j, pos in enumerate(posData):
            
            if types[j] == 1:
                costermA += np.cos(np.sum([kx, ky] * pos))
                sintermA += np.sin(np.sum([kx, ky] * pos))
            else:
                costermB += np.cos(np.sum([kx, ky] * pos))
                sintermB += np.sin(np.sum([kx, ky] * pos))

        SkAA[i] += (costermA**2 + sintermA**2) / 650
        SkBB[i] += (costermB**2 + sintermB**2) / 350

    SkAAn, SkBBn = np.zeros(Nuk), np.zeros(Nuk)
    for ik in range(Nuk):
        ki   = uk[ik]
        indk = np.nonzero(abs((knorm-ki)) < thresh)[0]
        SkAAn[ik] = sum(SkAA[indk])/len(indk)
        SkBBn[ik] = sum(SkBB[indk])/len(indk)
    
    return SkAAn, SkBBn, uk


def rdf_poi(rdf):
    """
    calculates all the points of interest for the radial distribution function
    """

    features = [[], [], [], [], [], [], []]


    for gr in rdf:
        grAA, grBB, grAB = gr[0], gr[1], gr[2]


        # height of the first peak of grAA
        features[0].append(max(grAA))

        # height of the second peak of grAA
        left = round(1.5 / params['dr'])
        right = round(2.2 / params['dr'])
        features[1].append(max(grAA[left:right]))

        # height of the fourth peak of grAA
        left = round(2.5 / params['dr'])
        right = round(3.5 / params['dr'])
        features[2].append(max(grAA[left:right]))

        # height of the second valley of grBB
        left = round(1.5 / params['dr'])
        right = round(2.5 / params['dr'])
        features[3].append(min(grBB[left:right]))

        # height of the second 'big' peak of grBB
        left = round(2.2 / params['dr'])
        right = round(3.3 / params['dr'])
        features[4].append(max(grBB[left:right]))

        # height of the first peak of grAB
        features[5].append(max(grAB))

        # height of the second peak of grAB
        left = round(1.5 / params['dr'])
        right = round(2.5 / params['dr'])
        features[6].append(max(grAB[left:right]))

    return features


def plot_young_old(timesteps1, timesteps2, feature1, feature2):
    """
    plots a young and old feature side by side
    """

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

        
    ax1.plot(timesteps1, feature1)
    ax2.plot(timesteps2, feature2)

    ax1.get_shared_y_axes().join(ax1, ax2)
    ax1.set_yticklabels([])

    plt.show()
