from dataread import read_data
from util import *
from visualise import *
from ML import *
import time
import matplotlib.pyplot as plt


Linear_regression = False
Binary_classification = True

# initialization
particles  = 1000
dimensions = 2
dt = 0.0001
iterations = 5000000
dump_interval = 1000

# get the positions of each particle for each timestep
timesteps, Data = read_data('traj_dump.atom', particles, dimensions, dt, iterations, dump_interval)

if Binary_classification:
    # take the first and last 20 percent of the data for the binary classifier
    m = int(0.2*len(timesteps))
    timesteps = np.concatenate(np.zeros(m), np.ones(m))

    # convert data
    Data_binary = np.array(list(Data.values()))
    Data = np.concatenate(Data_binary[:, :m], Data_binary[:, m:])
    Data = {'position': Data[0], 'force': Data[1], 'angMom': Data[2], 'torque': Data[3]}

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

if Linear_regression:
    linear_regression(features, timesteps, test_ratio=0.2)

if Binary_classification:
    logistic_regression(features, timesteps, test_ratio=0.2)
    
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

# Binary classifier

