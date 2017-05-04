import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from definitions import SAVE_PATH_FIGURES, SAVE_PATH_DATA, DATA_PATH
import scipy.io
import scipy.signal


data_folder = os.path.join(DATA_PATH, 'STN_data_PAC', '_selection')
file_list = os.listdir(data_folder)
save_folder = os.path.join(os.path.join(DATA_PATH, 'STN_data_PAC', 'raw'))

subject_file_list = [file for file in file_list if file.startswith('data') and file.endswith('.mat')]

data_dict = dict()

for file_idx, file in enumerate(subject_file_list):
    print('Analysing subject file', file)
    d = scipy.io.loadmat(os.path.join(data_folder, file))
    fs = d['fsample'][0][0]

    n_channels, n_samples, n_epochs = d['data'].shape

    data_dict['subject_{}'.format(file_idx)] = dict()

    for channel_idx, lfp_epochs in enumerate(d['data']):
        lfp_raw = lfp_epochs
        channel_label = d['channels'][:][channel_idx][0][0]
        print('Channel', channel_label)

        # filter in low and high beta band
        bands = [[11, 22], [18, 32]]
        for band_idx, band in enumerate(bands):
            print('band', band)
            esr_array = np.zeros(n_epochs)
            rdsr_array = np.zeros(n_epochs)
            # for every epoch
            for epoch_idx, epoch in enumerate(lfp_raw.T):
                # do preprocessing a la Cole et al
                lfp_pre = ut.low_pass_filter(y=epoch, fs=fs, cutoff=200)
                lfp_pre -= lfp_pre.mean()

                # band pass filter
                lfp_band = ut.band_pass_filter(y=lfp_pre, fs=fs, band=band, plot_response=False)
                # remove potential ringing artifacts
                idx_167ms = int((fs / 1000) * 167)
                lfp_band = lfp_band[idx_167ms:-idx_167ms]
                lfp_band -= lfp_band.mean()

                # calculate the sharpness and steepness ratios
                esr_array[epoch_idx], rdsr_array[epoch_idx] = ut.calculate_cole_ratios(lfp_pre, lfp_band, fs)

            # average over epochs
            esr = esr_array.mean()
            rdsr = rdsr_array.mean()

            # print(esr_array)
        break
    break
