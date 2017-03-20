import python.utils as ut
import matplotlib.pyplot as plt
import numpy as np
from definitions import SAVE_PATH_FIGURES, DATA_PATH
import scipy.signal
import os

"""
Calculate coherence between all contact pair combinations of contact from GPi left and right
"""

file_list = os.listdir(DATA_PATH)
suffix = ''
files_to_process = []
# select files
for f in file_list:
    if f.startswith('spmeeg_') and not f.startswith('spmeed_10'):
        files_to_process.append(f)

# set params
window_length = 1024
n_frequ_samples = int(window_length / 2 + 1)
n_patients = 27
n_electrodes = 153

list_right = ['GPiR01', 'GPiR12', 'GPiR23']
list_left = ['GPiL01', 'GPiL12', 'GPiL23']

# run analysis
for f in files_to_process:
    d = ut.load_data_spm(f)
    chanlabels = d['chanlabels'][0]
    fs = d['fsample'][0][0]

    plt.figure(figsize=(10, 7))

    # iterate over contact pair combinations left right
    plot_idx = 1
    for left_channel in list_left:
        for right_channel in list_right:
            # get a mask for the current channel pair
            mask_left = chanlabels == left_channel
            mask_right = chanlabels == right_channel

            # select the channels
            x = d['data'][mask_left, ].squeeze()
            y = d['data'][mask_right, ].squeeze()

            # calculate coherence
            f, coh = scipy.signal.coherence(x, y, fs=fs, window='hamming', nperseg=window_length)

            # plot
            mask = ut.get_array_mask(f > 4, f < 30)
            plt.subplot(3, 3, plot_idx)
            plt.plot(f[mask], np.real(coh)[mask], label='COH')

            f, coh = ut.coherence(x, y, fs=fs)
            plt.plot(f[mask], np.real(coh)[mask], label='myCOH')
            # plt.plot(f[mask], np.imag(coh).squeeze()[mask], label='iCOH')
            plt.legend()
            plot_idx += 1
    plt.show()
    break
