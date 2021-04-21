from dataread import read_data
from util import *
from visualise import *
from ML import *
import matplotlib.pyplot as plt
from parameters import *


#------------------------ INITIALIZATION ---------------------------

Linear_regression = False
Binary_classification = True
Visualisation = False

iterations = 100000
dump_interval = 1000

#------------------------ DATA PREPARATION --------------------------

if Linear_regression:
    # get the positions of each particle for each timestep
    timesteps, types, Data = read_data('traj_dump100000.atom', iterations, dump_interval)

if Binary_classification:
    
    timesteps1, types1, Data1 = read_data('traj_dump_young.atom', 50000, 50)
    timesteps2, types2, Data2 = read_data('traj_dump_old.atom', 1000000, 1000)

    # create binary classes for young and old
    timesteps1[timesteps1] = 0
    timesteps2[timesteps2] = 1

    timesteps = np.concatenate((timesteps1, timesteps2))
    types = np.concatenate((types1, types2))
    Data = np.concatenate((Data1, Data2), axis=1)


# convert to dictionary
Data = {'position': Data[0], 'force': Data[1], 'angMom': Data[2], 'torque': Data[3]}


#------------------------ FEATURE EXTRACTION -------------------------

# get the mean square displacement and the variance square displacement of the position data (deprecated)
# msd = msd(Data['position'])
# vsd = vsd(Data['position'], msd)

# calculate the mean and variance nearest neighbour distance per timestep
mnn_distance, mnn_amount = mean_nn(Data['position'], 1)
vnn_distance, vnn_amount  = variance_nn(Data['position'], mnn_distance, mnn_amount, 1)

# calculate the mean and variance of the norm of the force
mean_force = calc_mean(Data['force'])
variance_force = calc_variance(Data['force'], mean_force)

grAA, grBB, grAB = calc_rdf_peaks(Data['position'], types)

# prepare features in single array
features = np.column_stack([mnn_distance, vnn_distance, mean_force, variance_force, mnn_amount, vnn_amount, grAA, grBB])

#------------------------ PREDICTION ------------------------------

test_ratio = 0.2

if Linear_regression:
    # second degree polynomial regression
    linear_regression(features, timesteps, test_ratio, 2)
    # third degree polynomial regression
    linear_regression(features, timesteps, test_ratio, 3)

if Binary_classification:
    logistic_regression(features, timesteps, test_ratio)
    
if Visualisation:
    visualise(timesteps, Mean_square_displacement=msd,
                        Variance_square_displacement=vsd, 
                        Mean_nearest_neighbour_distance=mnn_distance, 
                        Variance_nearest_neighbour_distance=vnn_distance, 
                        Mean_force=mean_force, 
                        Variance_force=variance_force,
                        Mean_nearest_neighbour_amount=mnn_amount,
                        Variance_nearest_neighbour_amount=vnn_amount
                        )
