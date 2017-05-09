import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from definitions import SAVE_PATH_FIGURES, SAVE_PATH_DATA, DATA_PATH
import scipy.io
import scipy.signal

"""
The matlab data is saved in files for every subject condition. one subject condition file contains channels from both 
hemispheres. The PAC analysis results from Bernadette's analysis are saved in separate files in the same manner. 
This script collects the lfp data and the pac results of one subject into a single dictionary and pickles it.  
"""


data_folder = os.path.join(DATA_PATH, 'STN_data_PAC')
save_folder = os.path.join(DATA_PATH, 'STN_data_PAC', 'collected')
file_list = os.listdir(data_folder)

subject_list = ['DF', 'DP', 'JA', 'JB', 'DS', 'JN', 'JP', 'LM', 'MC', 'MW', 'SW', 'WB']

for subject_id in subject_list:

    subject_file_list = [file for file in file_list if subject_id in file and file.endswith('.mat')]

    super_dict = dict()

    # for every subject there should 4 files
    for file_idx, file in enumerate(subject_file_list):
        super_dict[file] = scipy.io.loadmat(os.path.join(data_folder, file))

    # collect data
    lfp_dict = dict(off=super_dict['data_{}_OFF.mat'.format(subject_id)],
                    on=super_dict['data_{}_ON.mat'.format(subject_id)])
    fs = lfp_dict['on']['fsample'][0][0]
    pac_dict = dict(off=super_dict['PAC_{}_OFF.mat'.format(subject_id)],
                    on=super_dict['PAC_{}_ON.mat'.format(subject_id)])

    # extract the left and right hemisphere channels
    # channel labels will be same within a subject
    channel_labels = np.squeeze(lfp_dict['on']['channels'])
    channel_labels = [chan[0] for chan in channel_labels]
    right_channels = [chan for chan in channel_labels if chan.startswith('STN_R')]
    left_channels = [chan for chan in channel_labels if chan.startswith('STN_L')]
    left_channel_idx = [channel_labels.index(lc) for lc in left_channels]
    right_channel_idx = [channel_labels.index(rc) for rc in right_channels]

    # now extract the PAC data
    f_amp = pac_dict['on']['F_amp'].squeeze()
    f_phase = pac_dict['on']['F_phase'].squeeze()
    conditions = ['off', 'on']
    n_conditions = len(conditions)

    # the frequency resolution should be the same for conditions
    n_amplitude = pac_dict['on']['F_amp'].size
    n_phase = pac_dict['on']['F_phase'].size

    # prelocat the pac matrix over all channels because we have to iterate over conditions first
    pac_matrix = np.zeros((len(channel_labels), n_conditions, n_amplitude, n_phase))

    # iterate over conditions
    for condition_idx, condition in enumerate(conditions):
        # get the data dict of the current condition
        condition_dict = pac_dict[condition]

        # extract data only for the current hemisphere channels
        for channel_idx, channel_string in enumerate(channel_labels):
            channel_key = 'PAC_{}'.format(channel_string)
            pac_matrix[channel_idx, condition_idx,] = condition_dict[channel_key]

    # look at the pac data and select useful frequency ranges for every condition, hemisphere and mean channel
    # look at left and right values separately
    pac_left = pac_matrix[left_channel_idx, ].mean(axis=(0, 2))  # mean over channels within a hemisphere
    pac_right = pac_matrix[right_channel_idx, ].mean(axis=(0, 2))  # , and over amp f?
    # new shape: (phase-f)

    # goal: find bands for every hemisphere
    bands = dict(left=[], right=[])
    for condition_idx, condition in enumerate(conditions):

        # for left
        # consider reasonable beta range
        mask = ut.get_array_mask(f_phase > 10, f_phase < 40).squeeze()
        f_mask = f_phase[mask]
        data = pac_left[condition_idx, mask]
        # smooth the mean PAC
        smoother_pac_left = ut.smooth_with_mean_window(data, window_size=5)
        max_idx = np.argmax(smoother_pac_left)

        # plt.subplot(1, 2, 1)
        # plt.title('Subject {}, {}'.format(subject_id, 'left'))
        # plt.plot(f_mask, data, alpha=.5)
        # plt.plot(f_mask, smoother_pac_left, label=condition)
        # plt.plot(f_mask[max_idx], smoother_pac_left[max_idx], 'ro')
        # plt.legend()
        # select the band +-5 Hz around the peak
        bands['left'].append([f_mask[max_idx]-5, f_mask[max_idx] + 5])

        # for right
        data = pac_right[condition_idx, mask]
        smoother_pac_right = ut.smooth_with_mean_window(data)
        max_idx = np.argmax(smoother_pac_right)

        # plt.subplot(1, 2, 2)
        # plt.title('Subject {}, {}'.format(subject_id, 'right'))
        # plt.plot(f_mask, data, alpha=.5)
        # plt.plot(f_mask, smoother_pac_right, label=condition)
        # plt.plot(f_mask[max_idx], smoother_pac_right[max_idx], 'ro', markersize=5)
        # plt.legend()
        # select the band +-5 Hz around the peak
        bands['right'].append([f_mask[max_idx] - 5, f_mask[max_idx] + 5])
    # plt.show()

    # save them in a dict and save the dict to disk
    subject_dict = dict(lfp=lfp_dict, pac=pac_dict, pac_matrix=pac_matrix, id=subject_id, fs=fs, bands=bands)
    print(bands)
    file_name = 'subject_{}_lfp_and_pac.p'.format(subject_id)
    ut.save_data(subject_dict, file_name, save_folder)