from dataread import read_data
from util import *
from visualise import *
from ML import *
import time
import matplotlib.pyplot as plt
import pandas as pd


# initialization
particles  = 1000
dimensions = 2
dt = 0.0001
iterations = 100000
dump_interval = 1000

# get the positions of each particle for each timestep
timesteps, Data = read_data('traj_dump.atom', particles, dimensions, dt, iterations, dump_interval)

# get the mean square displacement and the variance square displacement of the position data
msd = msd(Data['position'])
vsd = vsd(Data['position'], msd)

# calculate the mean and variance nearest neighbour distance per timestep
mnn_distance = mnn_distance(Data['position'])
vnn_distance = vnn_distance(Data['position'], mnn_distance)

# calculate the mean and variance of the norm of the forces
mean_force = calc_mean(Data['force'])
variance_force = calc_mean(Data['force'])

features = np.column_stack([mnn_distance, vnn_distance, mean_force, variance_force])

linear_regression(features, timesteps, test_ratio=0.2)

# simple plot for the features
visualise(timesteps, Data)
