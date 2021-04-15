import numpy as np


def read_data(fileName, particles, dimensions, dt, iterations, dump_interval):
    """
    reads a .atom file and returns the particles 3 dimensional position
    per timestep in a numpy array
    """

    size = int((iterations / dump_interval) + 1)

    # array of timesteps in which the position of all atoms can be found
    posData = np.zeros((size, particles, dimensions))
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
            x = float(line[2])
            y = float(line[3])

            if dimensions == 3:
                z = float(line[4])
                posData[timestep, particleId] = np.array([x, y, z])
            else:
                posData[timestep, particleId] = np.array([x, y])

        i += 1

    return posData, timesteps
