import numpy as np
import pandas as pd
from init import *


def read_data(fileName, iterations, dump_interval):
    """
    reads a .atom file and returns the particles 3 dimensional position
    per timestep in a numpy array
    """

    size = int((iterations / dump_interval) + 1)

    # array of timesteps in which the position and forces of all atoms can be found
    posData = np.zeros((size, params['particles'], params['dimensions']))
    forceData = np.zeros((size, params['particles'], params['dimensions']))
    q6_Re = np.zeros((size, params['particles'], 1))
    q6_Im = np.zeros((size, params['particles'], 1))
    
    vor_area = np.zeros((size, params['particles'], 1))
    vor_amn = np.zeros((size, params['particles'], 1))


    timesteps = np.arange(0, size)
    types = np.zeros((size, params['particles']))


    f = open(fileName, "r")
    i = 0

    for line in f:

        # new timestep
        if 'ITEM: TIMESTEP' in line:
            i = 0

        # get the timestep value
        if i == 1:
            timestep = int(int(line) / dump_interval)

        # skip first 8 lines for each timestep
        if i > 8:

            line = line.split(' ')

            particleId = int(line[0]) - 1
            types[timestep, particleId] = int(line[1])

            # position, force, dipole moment orientation, dipole moment magnitude, charge, angular momentum, torque
            position = np.array([line[2 + i] for i in range(params['dimensions'])])
            force = np.array([line[2 + params['dimensions'] + i] for i in range(params['dimensions'])])
            qr_value = np.array([line[2 + 2*params['dimensions']]])
            qi_value = np.array([line[3 + 2*params['dimensions']]])
            area = np.array([line[4 + 2*params['dimensions']]])
            amount = np.array([line[5 + 2*params['dimensions']]])

            posData[timestep, particleId] = position
            forceData[timestep, particleId] = force
            q6_Re[timestep, particleId] = qr_value
            q6_Im[timestep, particleId] = qi_value
            vor_area[timestep, particleId] = area
            vor_amn[timestep, particleId] = amount
            

        i += 1

    return timesteps, types, q6_Re, q6_Im, vor_area, vor_amn, np.array([posData, forceData])
