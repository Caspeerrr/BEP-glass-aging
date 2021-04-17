from dataread import read_data
from util import *
from visualise import *
from ML import *
import time
import matplotlib.pyplot as plt


# initialization
particles  = 1000
dimensions = 2
dt = 0.0001
iterations = 100000
dump_interval = 1000

# get the positions of each particle for each timestep
timesteps, Data = read_data('traj_dump100000.atom', particles, dimensions, dt, iterations, dump_interval)

# get the mean square displacement and the variance square displacement of the position data
msd = msd(Data['position'])
vsd = vsd(Data['position'], msd)

# calculate the mean and variance nearest neighbour distance per timestep
mnn_distance, mnn_amount = mean_nn(Data['position'], 1)

vnn_distance, vnn_amount  = variance_nn(Data['position'], mnn_distance, mnn_amount, 1)

# calculate the mean and variance of the norm of the force
mean_force = calc_mean(Data['force'])
variance_force = calc_variance(Data['force'], mean_force)

features = np.column_stack([mnn_distance, vnn_distance, mean_force, variance_force, mnn_amount, vnn_amount])

linear_regression(features, timesteps, test_ratio=0.33)

# simple plot for the features
visualise(timesteps, Mean_square_displacement=msd,
                     Variance_square_displacement=vsd, 
                     Mean_nearest_neighbour_distance=mnn_distance, 
                     Variance_nearest_neighbour_distance=vnn_distance, 
                     Mean_force=mean_force, 
                     Variance_force=variance_force,
                     Mean_nearest_neighbour_amount=mnn_amount,
                     Variance_nearest_neighbour_amount=vnn_amount
                     )
