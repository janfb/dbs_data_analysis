import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from definitions import SAVE_PATH_FIGURES, SAVE_PATH_DATA, DATA_PATH

data_folder = os.path.join(SAVE_PATH_DATA, 'stn')
save_folder = os.path.join(SAVE_PATH_FIGURES, 'stn')
filename = 'stn_data_sharpness_over_epochs.p'

data_dict = ut.load_data_analysis(filename, data_folder)
n_subject = len(data_dict.keys())
n_channels = 6

# make one big figure with channels subplot showing the esr and rdsr over condition for all subjects
plt.figure(figsize=(15, 8))

# for every subject, select the channel with the largest difference in esr
esr_over_conditions = np.zeros((n_subject, 2))
rdsr_over_conditions = np.zeros((n_subject, 2))

esr_over_channels = np.zeros((n_subject, 6, 2))
rdsr_over_channels = np.zeros((n_subject, 6, 2))
esr_over_channels_std = np.zeros((n_subject, 6, 2))
rdsr_over_channels_std = np.zeros((n_subject, 6, 2))

band_idx = 0
bands = [[11, 22], [18, 32]]
band_strings = ['low_beta', 'high_beta']

for subject_idx, subject_dict in enumerate(data_dict.values()):

    conditions = ['off', 'on']
    for condition_idx, condition in enumerate(conditions):
        condition_dict = subject_dict[condition]
        n_channels, n_bands, n_epochs = condition_dict['esr_matrix'].shape
        # extract over channels
        for channel_idx in range(n_channels):
            esr_over_channels[subject_idx, channel_idx, condition_idx] = condition_dict['esr_matrix'][channel_idx, band_idx, :].mean()
            esr_over_channels_std[subject_idx, channel_idx, condition_idx] = condition_dict['esr_matrix'][channel_idx,
                                                                             band_idx, :].std()
            rdsr_over_channels[subject_idx, channel_idx, condition_idx] = condition_dict['rdsr_matrix'][channel_idx,
                                                                          band_idx, :].mean()
            rdsr_over_channels_std[subject_idx, channel_idx, condition_idx] = condition_dict['rdsr_matrix'][channel_idx,
                                                                              band_idx,:].std()

# plt.show()
plt.close()

# plot the channel with maximal esr and rdsr

plt.figure(figsize=(12, 4))
condition_difference = abs(esr_over_channels[:, :, 0] - esr_over_channels[:, :, 1])
# select the channel with max difference in esr
channel_indices = np.argmax(condition_difference, axis=1)

for subject_idx in range(n_subject):
    plt.subplot(1, 2, 1)
    plt.plot(esr_over_channels[subject_idx, channel_indices[subject_idx], :], 'o-')
    plt.xticks(range(2), conditions)
    plt.ylabel('esr')
    plt.subplot(1, 2, 2)
    plt.plot(rdsr_over_channels[subject_idx, channel_indices[subject_idx], :], 'o-')
    plt.ylabel('rdsr')
    plt.xticks(range(2), conditions)

plt.suptitle('ESR and RDSR, channel with maximum difference, band = {}'.format(bands[band_idx]))
figure_filename = os.path.join(save_folder, 'sharpness_all_subjects_selected_channels_{}.pdf'.format(band_strings[band_idx]))
plt.savefig(figure_filename)
plt.show()
