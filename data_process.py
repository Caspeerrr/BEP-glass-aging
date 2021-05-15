from dataread import read_data
from util import *

def extract_features(directory, filename, iterations, dump_interval):
    
    timesteps, types, q6_re, q6_im, vor_area, vor_amn, Data = read_data(directory + filename, iterations, dump_interval)
    Data = {'position': Data[0], 'force': Data[1], 'q6_re': q6_re, 'q6_im': q6_im, 'vor_area': vor_area, 'vor_amn': vor_amn}

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

    # calculate the mean and variance of the real q6 parameters
    mean_q6_re = calc_mean(Data['q6_re'])
    variance_q6_re = calc_variance(Data['q6_re'], mean_q6_re)

    # calculate the mean and variance of the imaginary q6 parameters
    mean_q6_im = calc_mean(Data['q6_im'])
    variance_q6_im = calc_variance(Data['q6_im'], mean_q6_im)

    # calculate the mean and the variance of the voronoi area
    mean_vor_area = calc_mean(Data['vor_area'])
    variance_vor_area = calc_variance(Data['vor_area'], mean_vor_area)

    # calculate the mean and the variance of the voronoi area side amount
    mean_vor_amn = calc_mean(Data['vor_amn'])
    variance_vor_amn = calc_variance(Data['vor_amn'], mean_vor_amn)

    # calculate the first two peaks of the voronoi area and voronoi amount
    area_peak1_count, area_peak2_count,
    area_peak1_mag, area_peak2_mag, 
    amount_peak1_count, amount_peak2_count, 
    amount_peak1_mag, amount_peak2_mag = calc_voronoi_peaks(timesteps)

    # prepare features in single array
    features = np.column_stack([mnn_distance, vnn_distance, mean_force, variance_force,
                                mnn_amount, vnn_amount, mean_q6_re, variance_q6_re, mean_q6_im, 
                                variance_q6_im, mean_vor_area, variance_vor_area, 
                                mean_vor_amn, variance_vor_amn,
                                area_peak1_count, area_peak2_count,
                                area_peak1_mag, area_peak2_mag, 
                                amount_peak1_count, amount_peak2_count, 
                                amount_peak1_mag, amount_peak2_mag])

    return features
