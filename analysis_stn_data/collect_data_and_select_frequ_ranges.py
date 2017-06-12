import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from definitions import SAVE_PATH_FIGURES, SAVE_PATH_DATA, DATA_PATH
import scipy.io
import scipy.signal
from scipy.ndimage import measurements

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

for subject_id in subject_list:

    # get all mat files with the file with the subject ID
    subject_file_list = [file for file in file_list if subject_id in file and file.endswith('.mat')]

    super_dict = dict()

    # for every subject there should be 6 files: lfp, pac and significance-pac files for ON and OFF conditions.
    for file_idx, file in enumerate(subject_file_list):
        super_dict[file] = scipy.io.loadmat(os.path.join(data_folder, file))

    # collect data
    lfp_dict = dict(off=super_dict['data_{}_OFF.mat'.format(subject_id)],
                    on=super_dict['data_{}_ON.mat'.format(subject_id)])
    fs = lfp_dict['on']['fsample'][0][0]

    pac_dict = dict(off=super_dict['PAC_{}_OFF.mat'.format(subject_id)],
                    on=super_dict['PAC_{}_ON.mat'.format(subject_id)])

    sig_dict = dict(off=super_dict['signPAC_{}_OFF.mat'.format(subject_id)],
                    on=super_dict['signPAC_{}_ON.mat'.format(subject_id)])

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
    for channel_idx, channel_label in enumerate(channel_labels):

        # the customized freqeuncy bands are saved per hemisphere, therefore we have to find out the current hemi
        current_hemi = 'left' if channel_label in left_channels else 'right'

        # add a new list for condition bands of the current channel
        bands[current_hemi][channel_label] = []

        for condition_idx, condition in enumerate(conditions):

            # consider reasonable beta range
            mask = ut.get_array_mask(f_phase > 10, f_phase < 40).squeeze()
            f_mask = f_phase[mask]
            data = pac_phase[channel_idx, condition_idx, mask]
            # smooth the mean PAC
            smoother_pac = ut.smooth_with_mean_window(data, window_size=5)
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
            if max_cluster_size < 250:
                significant_pac[channel_idx, condition_idx] = 1

                # plot the PAC matrix for poster
                # ut.plot_sigcluster_illustration_for_poster(sig_matrix, pac_matrix, channel_idx, condition_idx, n_phase,
                #                                            n_amplitude,
                #                                            max_cluster_size)

            # plot both, the sig and the smoothed pac mean
            # plt.subplot(411)
            # plt.plot(current_sig_phase)
            # plt.subplot(412)
            # plt.plot(current_sig_amp)
            # plt.subplot(413)
            # plt.plot(smoother_pac)
            # plt.subplot(414)
            # plt.imshow(pac_matrix[channel_idx, condition_idx,], origin='lower')
            # plt.show()

            # plt.subplot(1, 2, 1)
            # plt.title('Subject {}, {}'.format(subject_id, 'left'))
            # plt.plot(f_mask, data, alpha=.5)
            # plt.plot(f_mask, smoother_pac_left, label=condition)
            # plt.plot(f_mask[max_idx], smoother_pac_left[max_idx], 'ro')
            # plt.legend()
            # select the band +-5 Hz around the peak

            bands[current_hemi][channel_label].append([f_mask[max_idx]- frequency_range_halflength,
                                                       f_mask[max_idx] + frequency_range_halflength])

    # save them in a dict and save the dict to disk
    subject_dict = dict(lfp=lfp_dict, pac=pac_dict, pac_matrix=pac_matrix, id=subject_id, fs=fs, bands=bands,
                        conditions=conditions, channel_labels=channel_labels, sig_flag_matrix=significant_pac)
    print(bands)
    file_name = 'subject_{}_lfp_and_pac.p'.format(subject_id)
    ut.save_data(subject_dict, file_name, save_folder)

plt.hist(np.array(max_cluster_list), bins='auto')
plt.show()