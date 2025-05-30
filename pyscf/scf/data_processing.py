'''
Dynamic Level Shifting (DLS) Helper Functions
'''

from scipy.signal import find_peaks
from sklearn import preprocessing as p
import numpy as np

def process_orbital_data(mo_energy, mo_occ): # Extract the relevant orbital data
    occupied_alpha = mo_energy[0][mo_occ[0] > 0]
    unoccupied_alpha = mo_energy[0][mo_occ[0] == 0]
    occupied_beta = mo_energy[1][mo_occ[1] > 0]
    unoccupied_beta = mo_energy[1][mo_occ[1] == 0]

    homo_alpha = occupied_alpha[-1] if len(occupied_alpha) > 0 else None
    lumo_alpha = unoccupied_alpha[0] if len(unoccupied_alpha) > 0 else None
    homo_1_beta = occupied_beta[-2] if len(occupied_beta) > 0 else None
    homo_beta = occupied_beta[-1] if len(occupied_beta) > 0 else None
    lumo_beta = unoccupied_beta[0] if len(unoccupied_beta) > 0 else None

    alpha_gap = lumo_alpha - homo_alpha

    return alpha_gap, homo_alpha, homo_1_beta, homo_beta

def find_num_peaks(series): # Determine the number of peaks in a time serie, including both  peaks and valleys
    series = np.array(series)
    peaks, _ = find_peaks(series)
    valleys, _ = find_peaks(-series)
    num_peaks_valleys = len(peaks) + len(valleys)
    return num_peaks_valleys

def generate_features(e_tot, alpha_homo, beta_homo_1, beta_homo, alpha_gap): # Generating the features required by the machine learning model
    e_peaks = find_num_peaks(e_tot) # Determine the oscilation in total energy
    a_homo_peaks = find_num_peaks(alpha_homo) # Determine the oscilation in alpha homo energy
    
    a_homo_med = np.median(alpha_homo)
    b_homo_1_med = np.median(beta_homo_1)
    b_homo_med = np.median(beta_homo)
    a_gap_med = np.median(alpha_gap)

    features = [e_peaks, a_homo_med, a_homo_peaks, b_homo_1_med, b_homo_med, a_gap_med]

    features_processed = np.array(features).reshape(1, -1)
    
    data_max = np.array([6.0, 0.10271725, 7.0, -0.03661817, 0.03391934, 0.33978315])
    data_min = [ 0.0, -0.25073079, 0.0, -0.53483508, -0.52753037, 0.01794255]

    normalizedData = (features_processed - data_min) / (data_max - data_min)

    return normalizedData