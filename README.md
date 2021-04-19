# Predicting the age of glass in active and passive matter using machine learning techniques

The outline of this repository is as follows:
- main.py <br>
calls all functions to print and visualize results

- dataread.py <br>
reads the data with following measured quantities: position, force, angular momentum and torque (in each direction) <br>
functions:

    * read_data: reads a .atom file and returns the timesteps and a dictionary where the keys are
the measures quantities and the values are the corresponding data of that feature

- util.py <br>
consists of various utilization functions used on the data:
    * msd: calculates the mean square displacement for each timestep
    * vsd: calculates the variance square displacement for each timestep
    * mnn_distance: calculates the mean nearest neighbour distance
    * vnn_distance: calculates the variance of the nearest neighbour distance
    * calc_mean: calculates the mean of the norm of the vector per timestep
    * calc_variance: calculates the variance of the norm of the vector per timestep

- visualise.py <br>
functions:

    * visualise: plots each measured quantity through time


- ML.py <br>
consists of various machine learning function used on the data:

    * linear_regression: predicts the age of a system with simple linear regression with regularization and returns this predictions
    * logistic_regression: binary classification method which predicts whether a system is young or old with regularization