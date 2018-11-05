import numpy as np
import os
import scipy.io
import sys

sys.path.append('../')
import utils as ut

"""
This script will calculate the three measures of nonsinusoidalness for a given list of subjects and beta bands. 
The beta bands are subject specific, e.g., every beta band corresponds to one specific subject. 

The three measures are calculated for all channels and conditions and saved in a matlab readable format. 
The specific channels and conditions can be selected from the saved matrix afterwards. 

Save it as a matlab struct with a field for every measure and then a field for every subject holding a matrix of 
dimensions: channel x condition

To save as struct one needs a dict in python: dict(esr={subject: matrix}, 
                                                   rdsr={subject: matrix},
                                                   plv={subject: matrix}) 
                                                   where matrix is channels x conditions x (mean, std), e.g., 6x2x2
"""

# path to the root of the repository. It should point to the folder with the data
save_folder = os.path.abspath(os.path.join('../../', 'analysis_data'))
data_folder = os.path.abspath(os.path.join('../../', 'data'))

input_file = scipy.io.loadmat('example_input.mat')

subject_list = [input_file['patient'][i][0][0] for i in range(len(input_file['patient']))]
channels = [input_file['channels'][i][0][0] for i in range(len(input_file['channels']))]
condition_list = [input_file['conditions'][i][0][0] for i in range(len(input_file['conditions']))]
beta_bands = list(map(list, list(input_file['beta_bands'])))
print(channels)
assert len(beta_bands) == len(subject_list)

# list of subject ids and list of corresponding beta bands read from a .mat file with a struct:

assert len(beta_bands) == len(subject_list)

# output
measures = dict(esr=dict(),
                rdsr=dict(),
                pvl=dict())

# define path to the data folder holding the STN data
file_list = os.listdir(data_folder)

# the outer loop is over different beta bands. for every band a new file will be saved

for subject_idx, subject_id in enumerate(subject_list):
    # iterate over subjects (in case there is more than one)

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
    esr_mat = np.zeros((n_channels, 2))  # last dim is for mean, std
    rdsr_mat = np.zeros((n_channels, 2))
    meanPhaseVec_mat = np.zeros((n_channels, 2))  # last to dims for amplitude and angle
    beta_amp_mat = np.zeros((n_channels))

    # get the data dict of the current condition
    condition_dict = lfp_dict[condition_list[subject_idx].lower()]

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

        esr_mat[channel_idx, :] = esr.mean(), esr.std()
        rdsr_mat[channel_idx, :] = rdsr.mean(), rdsr.std()
        meanPhaseVec_mat[channel_idx, :] = mpl.mean(), mpa.mean()
        beta_amp_mat[channel_idx] = np.mean(beta_amp)

    # save values for the current subject
    measures['esr'][subject_id] = esr_mat
    measures['rdsr'][subject_id] = rdsr_mat
    measures['pvl'][subject_id] = meanPhaseVec_mat

    # print results for testing
    print('Subject ID: {}, beta band {}, condition {}'.format(subject_id, beta_bands[subject_idx],
                                                              condition_list[subject_idx]))
    print('ESR {}'.format(esr_mat[:, 0]))
    print('RDSR {}'.format(rdsr_mat[:, 0]))
    print('PLV {}'.format(meanPhaseVec_mat[:, 0]))

# save results
scipy.io.savemat(os.path.join(save_folder, 'results_ESR_RDSR_PLV.mat'), measures)
