import numpy as np
import pandas as pd


def read_data(fileName, particles, dimensions, dt, iterations, dump_interval):
    """
    reads a .atom file and returns the particles 3 dimensional position
    per timestep in a numpy array
    """

    size = int((iterations / dump_interval) + 1)

    # array of timesteps in which the position and forces of all atoms can be found
    posData = np.zeros((size, particles, dimensions))
    forceData = np.zeros((size, particles, dimensions))
    dipMom_or = np.zeros((size, particles, dimensions))
    dipMom_mag = np.zeros((size, particles, 1))
    charge = np.zeros((size, particles, 1))
    angMom = np.zeros((size, particles, dimensions))
    torque = np.zeros((size, particles, dimensions))

    timesteps = np.arange(0, size)


    f = open(fileName, "r")
    i = 0

    for line in f:

        # new timestep
        if 'ITEM: TIMESTEP' in line:
            i = 0

        # get the timestep value
        if i == 1:
            timestep = int((int(line) - iterations) / dump_interval)            

        # skip first 8 lines for each timestep
        if i > 8:

            line = line.split(' ')

            particleId = int(line[0]) - 1

            # position, force, dipole moment orientation, dipole moment magnitude, charge, angular momentum, torque
            position = np.array([line[2 + i] for i in range(dimensions)])
            force = np.array([line[2 + dimensions + i] for i in range(dimensions)])
            mu_xy = np.array([line[2 + 2*dimensions + i] for i in range(dimensions)])
            mu_mag = np.array([line[2 + 3*dimensions]])
            q = np.array([line[3 + 3*dimensions]])
            L = np.array([line[4 + 3*dimensions + i] for i in range(dimensions)])
            T = np.array([line[4 + 4*dimensions + i] for i in range(dimensions)])

            posData[timestep, particleId] = position
            forceData[timestep, particleId] = force
            dipMom_or[timestep, particleId] = mu_xy
            dipMom_mag[timestep, particleId] = mu_mag
            charge[timestep, particleId] = q
            angMom[timestep, particleId] = L
            torque[timestep, particleId] = T

        i += 1


    return timesteps, {'position': posData, 
                       'force': forceData, 
                       'dipMom_or': dipMom_or,
                       'dipMom_mag': dipMom_mag,
                       'charge': charge,
                       'angMom': angMom, 
                       'torque': torque
                       }
