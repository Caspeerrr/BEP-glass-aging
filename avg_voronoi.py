from dataread import read_data
from util import *
from visualise import *
from ML import *
import matplotlib.pyplot as plt
from data_process import *
from init import *
import os
from progress.counter import Counter


def avg_voronoi(timestep):
    """
    Returns the voronoi area and the amount of voronoi edges for each particle 
    averaged over all dump files for given timestep
    """

    # initialization
    if timestep >= 0 and timestep <= 50000:
        iterations = 50000
        dump_interval = 50
        directory = ".\\dump\\young\\"
        extension = ".YOUNG"
    elif timestep >= 4000000 and timestep <= 5000000:
        timestep -= 4000000
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
            _, types, _, _, vor_area, vor_amn, Data = read_data(directory + filename, iterations, dump_interval)
            
            if amount == 0:
                voronoi_area = vor_area[timestep]
                voronoi_amount = vor_amn[timestep]
            else:
                voronoi_area += vor_area[timestep]
                voronoi_amount += vor_amn[timestep]
            
            amount += 1

    voronoi_area /= amount
    voronoi_amount /= amount

    return voronoi_area, voronoi_amount

def plot_voronoi(timestep):
    """
    Plots histograms of the voronoi area and the amount of voronoi edges
    for given timestep (averaged over all dump files)
    """
    
    voronoi_area, voronoi_amount = avg_voronoi(timestep)

    plt.hist(voronoi_area, bins=20)
    plt.title("Voronoi area")
    plt.show()

    plt.hist(voronoi_amount, bins=10)
    plt.title("Voronoi amount")
    plt.show()

plot_voronoi(50)