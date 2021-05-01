import os
from dataread import read_data
from util import *
from visualise import *
from ML import *
import matplotlib.pyplot as plt
from parameters import *

iterations = 5000000
dump_interval = 1000

directory = os.fsencode(".\\traj\\")
amount = 0
    
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".ATOM"): 
        timesteps, types, q6_re, q6_im, Data = read_data(".\\traj\\" + filename, iterations, dump_interval)

        if amount == 0:
            tmean_q6_re, tmean_q6_im = calc_mean(q6_re), calc_mean(q6_im)
            tvar_q6_re, tvar_q6_im = calc_variance(q6_re, tmean_q6_re), calc_variance(q6_im, tmean_q6_im)

        else:
            mean_q6_re, mean_q6_im = calc_mean(q6_re), calc_mean(q6_im)
            var_q6_re, var_q6_im = calc_variance(q6_re, mean_q6_re), calc_variance(q6_im, mean_q6_im)

            tmean_q6_re += mean_q6_re
            tmean_q6_im += mean_q6_im
            tvar_q6_re += var_q6_re
            tvar_q6_im += var_q6_im

        
        amount += 1

tmean_q6_re /= amount
tmean_q6_im /= amount
tvar_q6_re /= amount
tvar_q6_im /= amount

plt.plot(timesteps, tmean_q6_re)
plt.xlabel('timesteps')
plt.title('mean q6 real')
plt.show()
plt.plot(timesteps, tmean_q6_im)
plt.xlabel('timesteps')
plt.title('mean q6 imaginary')
plt.show()
plt.plot(timesteps, tvar_q6_re)
plt.xlabel('timesteps')
plt.title('variance q6 real')
plt.show()
plt.plot(timesteps, tvar_q6_im)
plt.xlabel('timesteps')
plt.title('variance q6 imaginary')
plt.show()

