from dataread import read_data
from util import *
from visualise import *
from ML import *
import matplotlib.pyplot as plt
from data_process import *
from init import *
import os
from progress.counter import Counter


def avg_rdf(timestep):

    # initialization
    if timestep > 0 and timestep < 50000:
        iterations = 50000
        dump_interval = 50
        directory = ".\\dump\\young\\"
        extension = ".YOUNG"
    elif timestep > 4000000 and timestep < 5000000:
        iterations = 1000000
        dump_interval = 1000
        directory = ".\\dump\\old\\"
        extension = ".OLD"
    else:
        raise ValueError("No data available for this timestep")

    if timestep % dump_interval != 0:
        raise ValueError("Timestep has to be dividible by the dump_interval")

    timestep = int(timestep / dump_interval)
    amount = 0
    counter = Counter('Calculating file ')

    # iterate through dump files
    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(extension): 

            counter.update()
            _, types, _, _, _, _, Data = read_data(directory + filename, iterations, dump_interval)
            
            if amount == 0:
                rdf = np.asarray(calc_rdf(Data[0, timestep], types[timestep]))
            else:
                rdf += np.asarray(calc_rdf(Data[0, timestep], types[timestep]))
            
            amount += 1

    rdf /= amount
    r = np.arange(0,params['rmax'] + params['dr'], params['dr'])
    plt.plot(r, rdf[0])
    plt.title(('average grAA at timestep', timestep))
    plt.show()
    plt.plot(r, rdf[1])
    plt.title(('average grBB at timestep', timestep))
    plt.show()
    plt.plot(r, rdf[2])
    plt.title(('average grAB at timestep', timestep))
    plt.show()

avg_rdf(50)
