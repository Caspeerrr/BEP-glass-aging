from dataread import read_data
from util import *

def extract_features(directory, filename, iterations, dump_interval):
    
    timesteps, types, q6_re, q6_im, Data = read_data(directory + filename, iterations, dump_interval)
    Data = {'position': Data[0], 'force': Data[1], 'q6_re': q6_re, 'q6_im': q6_im}


    savename = './/saves//' + filename.split(".")[0]

    # get the mean square displacement and the variance square displacement of the position data (deprecated)
    # msd = msd(Data['position'])
    # vsd = vsd(Data['position'], msd)

    # calculate the mean and variance nearest neighbour distance per timestep
    mnn_distance, mnn_amount = save_load(lambda: mean_nn(Data['position'], 1), savename + '-mean_nn.npy')
    vnn_distance, vnn_amount  = save_load(lambda: variance_nn(Data['position'], mnn_distance, mnn_amount, 1), savename + '-var_nn.npy')

    # calculate the mean and variance of the norm of the force
    mean_force = calc_mean(Data['force'])
    variance_force = calc_variance(Data['force'], mean_force)

    # prepare features in single array
    features = np.column_stack([mnn_distance, vnn_distance, mean_force, variance_force, mnn_amount, vnn_amount])

    return features
