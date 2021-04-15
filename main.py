from dataread import read_data
from util import *
from visualise import *
from ML import *
import time
import matplotlib.pyplot as plt


# initialization
particles = 1000
dimensions = 2
dt = 0.0001
iterations = 10000
dump_interval = 1000

# get the positions of each particle for each timestep
posData, timesteps = read_data('traj_dump1.atom', particles, dimensions, dt, iterations, dump_interval)

# get the mean square displacement and the variance square displacement of the position data
msd = msd(posData)
vsd = vsd(posData, msd)

print(msd)
print('\n')
print(vsd)

# calculate the mean and variance nearest neighbour distance per timestep
mnn_distance = mnn_distance(posData)
vnn_distance = vnn_distance(posData, mnn_distance)
print(mnn_distance, '\n')
print(vnn_distance)

features = np.column_stack([mnn_distance, vnn_distance])
linear_regression(features, timesteps)

visualise(timesteps, msd, vsd, mnn_distance, vnn_distance)
