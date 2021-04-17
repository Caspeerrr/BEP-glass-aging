# Predicting the age of glass in active and passive matter using machine learning techniques

The outline of this repository is as follows:
- main.py <br>
calls all functions to print and visualize results

- dataread.py <br>
reads the data with following measured quantities: position, force, angular momentum and torque (in each direction) <br>
functions:
<ul>
<li>read_data: reads a .atom file and returns the timesteps and a dictionary where the keys are
the measures quantities and the values are the corresponding data of that feature
</ul>

- util.py <br>
consists of various utilization functions used on the data:
<ul>
<li>msd: calculates the mean square displacement for each timestep
<li>vsd: calculates the variance square displacement for each timestep
<li>mnn_distance: calculates the mean nearest neighbour distance
<li>vnn_distance: calculates the variance of the nearest neighbour distance
<li>calc_mean: calculates the mean of the norm of the vector per timestep
<li>calc_variance: calculates the variance of the norm of the vector per timestep
</ul>

- visualise.py <br>
functions:
<ul>
<li>visualise: plots each measured quantity through time
</ul>

- ML.py <br>
consists of various machine learning function used on the data:
<ul>
<li>linear_regression: predicts the age of a system with simple linear regression without regularization and returns this predictions
</ul>