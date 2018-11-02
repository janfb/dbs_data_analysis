import numpy as np
import os
import scipy.io
import sys

from scipy.ndimage import measurements
sys.path.append('../')
import utils as ut

"""
This script will calculate the three measures of nonsinusoidalness for a given subject, channel, condition and beta 
frequency band. 
"""

# path to the root of the repository. CHANGE THIS PATH ACCORDINGLY. It should point to the folder with the data
root_path = '/Users/Jan/LRZBOX/Master/LR_Kuehn/'
data_folder = os.path.join(root_path, 'data', 'STN_data_PAC')


# list of subject ids to be included in the analysis
# subject_list = ['DF', 'DP', 'JA', 'JB', 'DS', 'JN', 'JP', 'LM', 'MC', 'MW', 'SW', 'WB']
subject_list = ['DF']
# list of channels to include
channel_numbers = ['STN_R01']
beta_bands = [[11, 22]]

assert len(subject_list) == len(channel_numbers) == len(beta_bands)

# output
measures = dict(esr=np.array([]), rdsr=np.array([]), pvl=np.array([]))

# define path to the data folder holding the STN data
file_list = os.listdir(data_folder)
conditions = ['off', 'on']
n_conditions = len(conditions)

# iterate over subjects (in case there is more than one)
for subject_idx, subject_id in enumerate(subject_list):

    # LOAD MAT FILES
    # get all mat files with the file with the subject ID
    subject_file_list = [file for file in file_list if subject_id in file and file.endswith('.mat')]

    super_dict = dict()
    # load mat files into dictionary
    for file_idx, file in enumerate(subject_file_list):
        # load matlab file as dict
        super_dict[file] = scipy.io.loadmat(os.path.join(data_folder, file))

    # EXTRACT LFP AND PAC DATA
    lfp_dict = dict(off=super_dict['data_{}_OFF.mat'.format(subject_id)],
                    on=super_dict['data_{}_ON.mat'.format(subject_id)])
    # get sampling rate
    fs = lfp_dict['on']['fsample'][0][0]

    pac_dict = dict(off=super_dict['PAC_{}_OFF.mat'.format(subject_id)],
                    on=super_dict['PAC_{}_ON.mat'.format(subject_id)])

    channel_labels = np.squeeze(lfp_dict['on']['channels'])
    channel_labels = [chan[0] for chan in channel_labels]
    n_channels = len(lfp_dict['on']['channels'])

    # prelocate arrays
    esr_mat = np.zeros((n_channels, n_conditions, 2))  # last dim is for mean, std
    rdsr_mat = np.zeros((n_channels, n_conditions, 2))
    meanPhaseVec_mat = np.zeros((n_channels, n_conditions, 2))  # last to dims for amplitude and angle
    beta_amp_mat = np.zeros((n_channels, n_conditions))

    # for every condition
    for condition_idx, condition in enumerate(conditions):
        # get the data dict of the current condition
        condition_dict = lfp_dict[condition]

        n_samples, n_epochs = condition_dict['data'].shape[1:]

        # select channels per hemisphere to treat them as separate subjects
        channel_labels_local = np.squeeze(condition_dict['channels'])

        for channel_idx, channel_label in enumerate(channel_labels_local):

            channel_label = channel_label[0]
            channel_lfp = condition_dict['data'][channel_idx]

            esr = np.zeros(n_epochs)
            rdsr = np.zeros(n_epochs)
            mpl = np.zeros(n_epochs)  # mean phase vector length
            mpa = np.zeros(n_epochs)  # amplitude
            beta_amp = np.zeros(n_epochs)

            # for every epoch
            for epoch_idx, epoch in enumerate(channel_lfp.T):
                # do preprocessing a la Cole et al
                # low pass filter
                lfp_pre = ut.low_pass_filter(y=epoch, fs=fs, cutoff=100, numtaps=250)

                # extract beta amplitude
                # calculate psd
                frequs, psd = ut.calculate_psd(lfp_pre, fs=fs, window_length=1024)
                # get the mask of the current band
                band_mask = ut.get_array_mask(frequs >= beta_bands[subject_idx][0], frequs <= beta_bands[subject_idx][1])
                # calculate beta amplitude
                beta_amp[epoch_idx] = np.mean(psd[band_mask])

                # band pass filter
                lfp_band = ut.band_pass_iir(y=lfp_pre, fs=fs, band=beta_bands[subject_idx])
                # remove potential ringing artifacts
                idx_167ms = int((fs / 1000) * 167)
                lfp_band = lfp_band[idx_167ms:-idx_167ms]
                lfp_band -= lfp_band.mean()

                lfp_pre = lfp_pre[idx_167ms: -idx_167ms]
                lfp_pre -= lfp_pre.mean()

                # calculate the sharpness and steepness ratios
                esr[epoch_idx], rdsr[epoch_idx] = ut.calculate_cole_ratios(lfp_pre, lfp_band, fs, epoch)
                mpl[epoch_idx], mpa[epoch_idx] = ut.calculate_mean_phase_amplitude(lfp_band, fs)

            esr_mat[channel_idx, condition_idx, :] = esr.mean(), esr.std()
            rdsr_mat[channel_idx, condition_idx, :] = rdsr.mean(), rdsr.std()
            meanPhaseVec_mat[channel_idx, condition_idx, :] = mpl.mean(), mpa.mean()
            beta_amp_mat[channel_idx, condition_idx] = np.mean(beta_amp)

    # print results for testing
    print('Subject ID: {}, Channel {}, beta band {}'.format(subject_id, channel_numbers[subject_idx], beta_bands[subject_idx]))
    print('ESR {}'.format(esr_mat[channel_labels.index(channel_numbers[subject_idx]), :, 0]))
    print('RDSR {}'.format(rdsr_mat[channel_labels.index(channel_numbers[subject_idx]), :, 0]))
    print('PLV {}'.format(meanPhaseVec_mat[channel_labels.index(channel_numbers[subject_idx]), :, 0]))