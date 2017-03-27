import os

import matplotlib.pyplot as plt
import numpy as np

import utils as ut
from definitions import SAVE_PATH_FIGURES, DATA_PATH

"""
Calculate coherence between all contact pair combinations of contact from GPi left and right
"""

plot_subjects = False

file_list = os.listdir(DATA_PATH)
suffix = ''
files_to_process = []
# select files
for f in file_list:
    if f.startswith('spmeeg_'):
        files_to_process.append(f)

# set params
window_length = 1024
n_frequ_samples = int(window_length / 2 + 1)
n_patients = len(files_to_process)
n_electrodes = 153

n_combinations = 9
coh_mat = np.empty((n_patients, n_combinations, n_frequ_samples))
icoh_mat = np.empty((n_patients, n_combinations, n_frequ_samples))

# run analysis
for f_idx, file in enumerate(files_to_process):
    print('Processing file {}'.format(file))
    d = ut.load_data_spm(file)
    chanlabels = d['chanlabels'][0]
    fs = d['fsample'][0][0]

    # extract the channels of this subject
    list_right = [label[0][:] for label in chanlabels if label[0][:].startswith('GPiR')]
    list_left = [label[0][:] for label in chanlabels if label[0][:].startswith('GPiL')]

    # only if there channels in both hemisphere, continue the analysis
    if len(list_left) and len(list_right):
        if plot_subjects:
            plt.figure(figsize=(10, 7))

        # iterate over contact pair combinations left right
        pair_idx = 0
        for left_channel in list_left:
            for right_channel in list_right:
                # get a mask for the current channel pair
                mask_left = chanlabels == left_channel
                mask_right = chanlabels == right_channel

                # select the channels
                x = d['data'][mask_left, ].squeeze()
                y = d['data'][mask_right, ].squeeze()

                # calculate coherency
                f, cohy = ut.coherency(x, y, fs=fs)

                coh, icoh = np.real(cohy), np.imag(cohy)
                coh_mat[f_idx, pair_idx, :] = coh
                icoh_mat[f_idx, pair_idx, :] = icoh

                # plot
                if plot_subjects:
                    mask = ut.get_array_mask(f > 4, f < 30)
                    plt.subplot(3, 3, pair_idx + 1)
                    plt.plot(f[mask], coh[mask], label='COH')
                    plt.plot(f[mask], icoh[mask], label='iCOH')
                    plt.title('{} - {}'.format(left_channel, right_channel))
                    if not pair_idx:
                        plt.legend()
                    if not pair_idx % 3:
                        plt.ylabel('coherence')
                    if pair_idx > 5:
                        plt.xlabel('Frequency [Hz]')
                    else:
                        plt.xticks([], [])
                pair_idx += 1
        # plt.show()
        if plot_subjects:
            plt.suptitle('Coherence and imaginary coherency for subject {}'.format(f_idx + 1))
            plt.savefig(os.path.join(SAVE_PATH_FIGURES, 'coherence', file[:-4] + '_coh{}.pdf'.format(suffix)))
            # plt.show()
            plt.close()

# save data
save_dict = dict(file_list=files_to_process, window_length=window_length, coh_mat=coh_mat, icoh_mat=icoh_mat, frequs=f)
ut.save_data(save_dict, filename='coh_icoh_allsubjects_w{}_{}.p'.format(window_length, suffix))
