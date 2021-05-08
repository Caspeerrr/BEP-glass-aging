from dataread import read_data
from util import *
from visualise import *
from ML import *
import matplotlib.pyplot as plt
from data_process import *
from parameters import *
import os


#------------------------ INITIALIZATION ---------------------------

Linear_regression = False
Binary_classification = True
Visualisation = False

iterations = 100000
dump_interval = 1000

#------------------------ DATA PREPARATION --------------------------

features = []
timesteps = []

if Binary_classification:
    # iterate through all dump files    
    for file in os.listdir(os.fsencode(".\\dump\\young\\")):
        filename = os.fsdecode(file)
        if filename.endswith(".YOUNG"): 

            featuresNew = extract_features(".\\dump\\young\\", filename, 50000, 50)[0]
            timestepsNew = np.zeros(len(featuresNew))

            features.append(featuresNew)
            timesteps.extend(timestepsNew)

    # iterate through all dump files    
    for file in os.listdir(os.fsencode(".\\dump\\old\\")):
        filename = os.fsdecode(file)
        if filename.endswith(".OLD"): 

            featuresNew = extract_features(".\\dump\\old\\", filename, 1000000, 1000)[0]
            timestepsNew = np.ones(len(featuresNew))

            features.append(featuresNew)
            timesteps.extend(timestepsNew)


if Linear_regression:

    # iterate through all dump files    
    for file in os.listdir(os.fsencode(".\\dump\\full\\")):
        filename = os.fsdecode(file)
        if filename.endswith(".ATOM"): 

            featuresNew = extract_features(".\\dump\\full\\", filename, 5000000, 1000)[0]
            timestepsNew = np.zeros(len(featuresNew))

            features.append(featuresNew)
            timesteps.extend(timestepsNew)

#------------------------ PREDICTION ------------------------------

if Linear_regression:
    # second degree polynomial regression
    linear_regression(features, timesteps, params['test_ratio'], degree=2)
    # third degree polynomial regression
    linear_regression(features, timesteps, params['test_ratio'], degree=3)

if Binary_classification:
    logistic_regression(features, timesteps, params['test_ratio'])
    
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
