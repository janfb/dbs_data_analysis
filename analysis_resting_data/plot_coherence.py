import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from definitions import SAVE_PATH_FIGURES

"""
Plot the mean coherence over subjects. select for every subject the channel with the largest amplitude in coherence. 
"""

# load data
suffix = ''
window_length = 1024
data = ut.load_data_analysis('coh_icoh_allsubjects_w{}_{}.p'.format(window_length, suffix))
# now make a plot of all selected channels and the histogram over peaks
f = data['frequs']
mask = ut.get_array_mask(f > 1, f < 30)
coh_mat = data['coh_mat'][:, :, mask]
icoh_mat = data['icoh_mat'][:, :, mask]
n_subjects, n_channels, n_frequ_samples = coh_mat.shape

max_coh_channels = np.zeros((n_subjects, n_frequ_samples))
max_icoh_channels = np.zeros((n_subjects, n_frequ_samples))

for subject_idx in range(n_subjects):
    # select the channel with the largest coherence
    channel_idx = np.argmax(np.max(coh_mat[subject_idx, ], axis=1))

    max_coh_channels[subject_idx, ] = coh_mat[subject_idx, channel_idx, :]
    max_icoh_channels[subject_idx, ] = icoh_mat[subject_idx, channel_idx, :]

# take mean and SE over subjects
mean = np.mean(max_coh_channels, axis=0)
se = ut.standard_error(max_coh_channels, axis=0)

imean = np.mean(max_icoh_channels, axis=0)
ise = ut.standard_error(max_icoh_channels, axis=0)

plt.figure(figsize=(7, 5))
plt.plot(f[mask], mean, label='mean COH')
plt.fill_between(f[mask], mean + se, mean - se, alpha=.2, color='C0', label='+-standard error')
plt.plot(f[mask], imean, label='mean iCOH')
plt.fill_between(f[mask], imean + ise, imean - ise, alpha=.2, color='C1', label='+-standard error')

plt.title('Interhemispheric coherence and imaginary coherency')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Coherence')
plt.legend()
plt.savefig(os.path.join(SAVE_PATH_FIGURES, 'coherence', 'figure4A_{}.pdf'.format(suffix)))
# plt.show()
plt.close()
