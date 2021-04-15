import numpy as np


def read_data(fileName, particles, dimensions, dt, iterations, dump_interval):
    """
    reads a .atom file and returns the particles 3 dimensional position
    per timestep in a numpy array
    """

    size = int((iterations / dump_interval) + 1)

    # array of timesteps in which the position and forces of all atoms can be found
    posData = np.zeros((size, particles, dimensions))
    forceData = np.zeros((size, particles, dimensions))
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
            position = np.array([line[2 + i] for i in range(dimensions)])
            force = np.array([line[2 + dimensions + i] for i in range(dimensions)])

            posData[timestep, particleId] = position
            forceData[timestep, particleId] = force

        i += 1

    return posData, forceData, timesteps
