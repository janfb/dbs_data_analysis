import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from definitions import SAVE_PATH_FIGURES, SAVE_PATH_DATA, DATA_PATH
import scipy.io
import scipy.signal
from scipy.ndimage import measurements
import analysis_stn_data.plotting_functions as plotter

"""
The matlab data is saved in files for every subject condition. one subject condition file contains channels from both 
hemispheres. The PAC analysis results from Bernadette's analysis are saved in separate files in the same manner. 
This script collects the lfp data and the pac results of one subject into a single dictionary and pickles it.  
"""


data_folder = os.path.join(DATA_PATH, 'STN_data_PAC')
save_folder = os.path.join(DATA_PATH, 'STN_data_PAC', 'collected')
file_list = os.listdir(data_folder)

subject_list = ['DF', 'DP', 'JA', 'JB', 'DS', 'JN', 'JP', 'LM', 'MC', 'MW', 'SW', 'WB']

max_cluster_list = []
# number of connected significant bins to call it a cluster
cluster_criterion = 150

for subject_id in subject_list:

    # get all mat files with the file with the subject ID
    subject_file_list = [file for file in file_list if subject_id in file and file.endswith('.mat')]

    super_dict = dict()

    # for every subject there should be 6 files: lfp, pac and significance-pac files for ON and OFF conditions.
    for file_idx, file in enumerate(subject_file_list):
        # load matlab file as dict
        super_dict[file] = scipy.io.loadmat(os.path.join(data_folder, file))

    # collect data
    lfp_dict = dict(off=super_dict['data_{}_OFF.mat'.format(subject_id)],
                    on=super_dict['data_{}_ON.mat'.format(subject_id)])
    # get sampling rate
    fs = lfp_dict['on']['fsample'][0][0]

    pac_dict = dict(off=super_dict['PAC_{}_OFF.mat'.format(subject_id)],
                    on=super_dict['PAC_{}_ON.mat'.format(subject_id)])

    sig_dict = dict(off=super_dict['signPAC_{}_OFF.mat'.format(subject_id)],
                    on=super_dict['signPAC_{}_ON.mat'.format(subject_id)])

    # extract the left and right hemisphere channels
    # channel labels will be same within a subject
    channel_labels = np.squeeze(lfp_dict['on']['channels'])
    # remove singleton dimension
    channel_labels = [chan[0] for chan in channel_labels]
    right_channels = [chan for chan in channel_labels if chan.startswith('STN_R')]
    left_channels = [chan for chan in channel_labels if chan.startswith('STN_L')]
    left_channel_idx = [channel_labels.index(lc) for lc in left_channels]
    right_channel_idx = [channel_labels.index(rc) for rc in right_channels]

    # extract the PAC data
    f_amp = pac_dict['on']['F_amp'].squeeze()
    f_phase = pac_dict['on']['F_phase'].squeeze()
    conditions = ['off', 'on']
    n_conditions = len(conditions)

    # the frequency resolution should be the same for conditions
    n_amplitude = pac_dict['on']['F_amp'].size
    n_phase = pac_dict['on']['F_phase'].size

    # prelocate the pac matrix over all channels because we have to iterate over conditions first
    pac_matrix = np.zeros((len(channel_labels), n_conditions, n_amplitude, n_phase))
    # and the significance matrix, a logical of the same shape indicatiing permutation test significance for every bin
    sig_matrix = np.zeros((len(channel_labels), n_conditions, n_amplitude, n_phase))

    # iterate over conditions
    for condition_idx, condition in enumerate(conditions):
        # get the data dict of the current condition
        pac_condition_dict = pac_dict[condition]
        sig_condition_dict = sig_dict[condition]

        # extract data only for the current hemisphere channels
        for channel_idx, channel_string in enumerate(channel_labels):
            pac_channel_key = 'PAC_{}'.format(channel_string)
            sig_channel_key = 'signPAC_{}'.format(channel_string)
            pac_matrix[channel_idx, condition_idx, ] = pac_condition_dict[pac_channel_key]
            sig_matrix[channel_idx, condition_idx, ] = sig_condition_dict[sig_channel_key]

    # look at the pac data and select useful frequency ranges for every condition, hemisphere and mean channel
    # look at left and right values separately
    pac_phase = pac_matrix.mean(axis=2)  # average over amplitude frequencies
    # new shape: (hemi_channels, conditions, phase-f)

    # goal: find bands for every hemisphere
    bands = dict(left=dict(), right=dict())
    frequency_range_halflength = 6
    # have a matrix for significance flag
    significant_pac = np.zeros((len(channel_labels), n_conditions))
    # set a threshold for the locigal mean over amplitude frequencies
    sig_threshold = 0.3

    # for every channel and condition in each condition, select a good PAC phase frequency range
    # bet band selection illustration plot
    for channel_idx, channel_label in enumerate(channel_labels):

        # the customized freqeuncy bands are saved per hemisphere, therefore we have to find the current hemi
        current_hemi = 'left' if channel_label in left_channels else 'right'

        # add a new list for condition bands of the current channel
        bands[current_hemi][channel_label] = []

        for condition_idx, condition in enumerate(conditions):

            # get current lfp data
            current_lfp_epochs = lfp_dict[condition]['data'][channel_idx]

            # consider reasonable beta range
            mask = ut.get_array_mask(f_phase >= 12, f_phase <= 40).squeeze()
            f_mask = f_phase[mask]
            data = pac_phase[channel_idx, condition_idx, mask]
            # smooth the mean PAC
            smoother_pac = ut.smooth_with_mean_window(data, window_size=3)
            max_idx = np.argmax(smoother_pac)
            # sum logical significance values across the amplitude frequency dimension
            # calculate the binary groups in the significance map
            lw, num = measurements.label(sig_matrix[channel_idx, condition_idx, : , :])
            # calculate the area of the clusters:
            # from http://stackoverflow.com/questions/25664682/how-to-find-cluster-sizes-in-2d-numpy-array
            area = measurements.sum(sig_matrix[channel_idx, condition_idx,], lw, index=np.arange(lw.max() + 1))
            # get the size of the largest group
            max_cluster_size = np.max(area)
            max_cluster_list.append(max_cluster_size)
            # calculate mean
            current_sig_phase = sig_matrix[channel_idx, condition_idx, :, mask].mean(axis=1)  # should be shape (61,)
            current_sig_amp = sig_matrix[channel_idx, condition_idx, :, mask].mean(axis=0)  # should be shape (61,)
            # if (np.max(current_sig_phase) > sig_threshold or np.max(current_sig_amp) > sig_threshold) \
            if max_cluster_size > cluster_criterion:
                significant_pac[channel_idx, condition_idx] = 1

            # plotter.plot_beta_band_selection_illustration_for_poster(pac_matrix[channel_idx, condition_idx,],
            #                                                          pac_matrix[channel_idx, condition_idx,],
            #                                                          sig_matrix[channel_idx, condition_idx,],
            #                                                          sig_matrix[channel_idx, condition_idx,], n_phase,
            #                                                          n_amplitude,
            #                                                          f_phase, f_amp, mask, smoother_pac, max_idx,
            #                                                          current_lfp_epochs,
            #                                                          subject_id, fs)

            # select the band +-5 Hz around the peak
            bands[current_hemi][channel_label].append([f_mask[max_idx]- frequency_range_halflength,
                                                       f_mask[max_idx] + frequency_range_halflength])

    # save them in a dict and save the dict to disk
    subject_dict = dict(lfp=lfp_dict, pac=pac_dict, sig_matrix=sig_matrix, pac_matrix=pac_matrix, id=subject_id, fs=fs,
                        bands=bands,
                        conditions=conditions, channel_labels=channel_labels, sig_flag_matrix=significant_pac)
    print(bands)
    file_name = 'subject_{}_lfp_and_pac.p'.format(subject_id)
    ut.save_data(subject_dict, file_name, save_folder)

print('data_points:', np.sum(np.array(max_cluster_list) > cluster_criterion))
# plt.hist(np.array(max_cluster_list), bins='auto')
# plt.show()