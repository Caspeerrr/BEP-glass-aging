import os
from dataread import read_data
from util import *
from visualise import *
from ML import *
import matplotlib.pyplot as plt
from parameters import *


iterations = 50000
dump_interval = 50

rdf = True
q6 = True
directory = ".\\dump\\old\\"
extension = ".OLD"

amount = 0

# iterate through all dump files    
for file in os.listdir(os.fsencode(directory)):
     filename = os.fsdecode(file)
     if filename.endswith(extension): 
        
        # retrieve and process data
        timesteps, types, q6_re, q6_im, Data = read_data(directory + filename, iterations, dump_interval)
        Data = {'position': Data[0], 'force': Data[1], 'q6_re': q6_re, 'q6_im': q6_im}

        # declare variable for the first run
        if amount == 0:
            
            if rdf:
                gr = calc_avg_rdf(Data['position'], types)

            if q6:
                mean_q6 = np.asarray((calc_mean(Data['q6_re']), calc_mean(Data['q6_im'])))
                var_q6 = np.asarray((calc_variance(Data['q6_re'], mean_q6[0]), calc_variance(Data['q6_im'], mean_q6[1])))

        else:
            if rdf:
                gr += np.asarray((calc_avg_rdf(Data['position'], types)))

            if q6:
                mean_q6 += np.asarray((calc_mean(Data['q6_re']), calc_mean(Data['q6_im'])))
                var_q6 += np.asarray((calc_variance(Data['q6_re'], mean_q6[0]), calc_variance(Data['q6_im'], mean_q6[1])))
        
        amount += 1

# visualise the found averages
if rdf:
    gr /= amount
    r = np.arange(0,rmax+dr,dr)
    plt.plot(r, gr[0])
    plt.title('grAA')
    plt.xlabel('r')
    plt.show()
    plt.plot(r, gr[1])
    plt.title('grBB')
    plt.xlabel('r')
    plt.show()
    plt.plot(r, gr[2])
    plt.title('grAB')
    plt.xlabel('r')
    plt.show()

if q6:
    mean_q6 /= amount
    var_q6 /= amount
    visualise(timesteps, False, q6_mean_real=mean_q6[0], q6_mean_imaginary=mean_q6[1],
                         q6_variance_real=var_q6[0], q6_variance_imaginary=var_q6[1])
