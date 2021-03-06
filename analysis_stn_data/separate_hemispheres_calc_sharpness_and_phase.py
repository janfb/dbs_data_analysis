import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from definitions import SAVE_PATH_FIGURES, SAVE_PATH_DATA, DATA_PATH
import scipy.io
import scipy.signal

"""
The script collect_data.py collects the data of one subject into one dictionary. This script collects the lfp data and 
pac results of one subject hemisphere and calculates the sharpness and slope steepness ratios. It saves the sharpness 
and pac results into one dictionary and saves this subject_hemisphere dictionary in a super 
dictionary with subject_{ID}_{left, right} indices.
"""


data_folder = os.path.join(DATA_PATH, 'STN_data_PAC', 'collected')
file_list = os.listdir(data_folder)
save_folder = os.path.join(SAVE_PATH_DATA, 'stn')

subject_file_list = [file for file in file_list if file.startswith('subject') and file.endswith('.p')]

data_dict = dict()

for file_idx, file in enumerate(subject_file_list):

    print('Analysing subject file', file)

    super_dict = np.load(os.path.join(data_folder, file))
    subject_id = super_dict['id']

    # collect data
    lfp_dict = super_dict['lfp']
    fs = super_dict['fs']
    pac_dict = super_dict['pac']

    # channel labels will be same within a subject
    channel_labels = np.squeeze(lfp_dict['on']['channels'])
    channel_labels = [chan[0] for chan in channel_labels]
    right_channels = [chan for chan in channel_labels if chan.startswith('STN_R')]
    left_channels = [chan for chan in channel_labels if chan.startswith('STN_L')]
    left_channel_idx = [channel_labels.index(lc) for lc in left_channels]
    right_channel_idx = [channel_labels.index(rc) for rc in right_channels]

    # LFP DATA
    # over conditions
    conditions = ['off', 'on']
    n_conditions = len(conditions)
    n_channels = len(lfp_dict['on']['channels'])
    bands = [[11, 22]]
    n_bands = len(bands)

    esr_mat = np.zeros((n_channels, n_bands, n_conditions, 2))  # last dim is for mean, std
    rdsr_mat = np.zeros((n_channels, n_bands, n_conditions, 2))
    meanPhaseVec_mat = np.zeros((n_channels, n_bands, n_conditions, 2))  # last to dims for amplitude and angle
    beta_amp_mat = np.zeros((n_channels, n_bands, n_conditions))

    for condition_idx, condition in enumerate(conditions):
        # get the data dict of the current condition
        condition_dict = lfp_dict[condition]

        n_samples, n_epochs = condition_dict['data'].shape[1:]

        # select channels per hemisphere to treat them as separate subjects
        channel_labels_local = np.squeeze(condition_dict['channels'])

        for channel_idx, channel_label in enumerate(channel_labels_local):

            channel_label = channel_label[0]
            channel_lfp = condition_dict['data'][channel_idx]

            # the customized freqeuncy bands are saved per hemisphere, therefore we have to find out the current hemi
            current_hemi = 'left' if channel_label in left_channels else 'right'
            # the frequency bands are also different for channels and conditions, pick the current one.
            current_band = super_dict['bands'][current_hemi][channel_label][condition_idx]

            # filter in low and high beta band
            for band_idx, band in enumerate(bands):

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
                    band_mask = ut.get_array_mask(frequs >= current_band[0], frequs <= current_band[1])
                    # calculate beta amplitude
                    beta_amp[epoch_idx] = np.mean(psd[band_mask])

                    # band pass filter
                    lfp_band = ut.band_pass_iir(y=lfp_pre, fs=fs, band=current_band)
                    # remove potential ringing artifacts
                    idx_167ms = int((fs / 1000) * 167)
                    lfp_band = lfp_band[idx_167ms:-idx_167ms]
                    lfp_band -= lfp_band.mean()

                    lfp_pre = lfp_pre[idx_167ms: -idx_167ms]
                    lfp_pre -= lfp_pre.mean()

                    # debug plot, check artifacts
                    # if subject_id == 'DP':
                    #     plt.plot(lfp_pre)
                    #     plt.show()

                    # calculate the sharpness and steepness ratios
                    # print(file_idx, condition_idx, channel_idx, epoch_idx)
                    esr[epoch_idx], rdsr[epoch_idx] = ut.calculate_cole_ratios(lfp_pre, lfp_band, fs, epoch)
                    mpl[epoch_idx], mpa[epoch_idx] = ut.calculate_mean_phase_amplitude(lfp_band, fs)

                esr_mat[channel_idx, band_idx, condition_idx, :] = esr.mean(), esr.std()
                rdsr_mat[channel_idx, band_idx, condition_idx, :] = rdsr.mean(), rdsr.std()
                meanPhaseVec_mat[channel_idx, band_idx, condition_idx, :] = mpl.mean(), mpa.mean()
                beta_amp_mat[channel_idx, band_idx, condition_idx] = np.mean(beta_amp)

    # now we have a matrix with channels x bands x conditions x (mean, std) for one subject

    # now extract the PAC data
    # the frequency resolution should be the same for conditions
    f_phase = pac_dict['on']['F_phase']
    f_amp = pac_dict['on']['F_amp']
    n_amplitude = f_amp.size
    n_phase = f_phase.size

    # prelocat the pac matrix over all channels because we have to iterate over conditions first
    pac_matrix = np.zeros((len(channel_labels), n_conditions, n_amplitude, n_phase))

    # iterate over conditions
    for condition_idx, condition in enumerate(conditions):
        # get the data dict of the current condition
        condition_dict = pac_dict[condition]

        # extract data only for the current hemisphere channels
        for channel_idx, channel_string in enumerate(channel_labels):

            channel_key = 'PAC_{}'.format(channel_string)
            pac_matrix[channel_idx, condition_idx, ] = condition_dict[channel_key]

    # we now separate the hemispheres into separate subjects
    # if subject is not in data_dict define sub dictionaries
    key_left = '{}_left'.format(subject_id)
    key_right = '{}_right'.format(subject_id)

    # select only left channels
    data_dict[key_left] = dict(esr_mat=esr_mat[left_channel_idx, ],
                               rdsr_mat=rdsr_mat[left_channel_idx, ],
                               pac_matrix=pac_matrix[left_channel_idx, ],
                               meanPhaseVec_mat=meanPhaseVec_mat[left_channel_idx, ],
                               beta_amp_mat=beta_amp_mat[left_channel_idx, ],
                               f_amp=pac_dict['on']['F_amp'],
                               f_phase=pac_dict['on']['F_phase'],
                               id=key_left,
                               conditions=conditions,
                               channel_labels=np.array(channel_labels)[left_channel_idx],
                               condition_bands=super_dict['bands']['left'],
                               significance_flag=super_dict['sig_flag_matrix'][left_channel_idx, ])

    # select only right channels
    data_dict[key_right] = dict(esr_mat=esr_mat[right_channel_idx,],
                                rdsr_mat=rdsr_mat[right_channel_idx,],
                                pac_matrix=pac_matrix[right_channel_idx, ],
                                meanPhaseVec_mat=meanPhaseVec_mat[right_channel_idx, ],
                                beta_amp_mat=beta_amp_mat[right_channel_idx, ],
                                f_amp=pac_dict['on']['F_amp'],
                                f_phase=pac_dict['on']['F_phase'],
                                id=key_right,
                                conditions=conditions,
                                channel_labels=np.array(channel_labels)[right_channel_idx],
                                condition_bands=super_dict['bands']['right'],
                                significance_flag=super_dict['sig_flag_matrix'][right_channel_idx, ])

# finally we save the big data file that contains the data for all hemispheres separately
save_filename = 'sharpness_pac_separated_in_hemisphere_n{}_optimized_bands.p'.format(file_idx + 1)
ut.save_data(data_dict, filename=save_filename, folder=save_folder)
