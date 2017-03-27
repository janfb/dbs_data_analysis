import os

import matplotlib.pyplot as plt
import numpy as np

import utils as ut
from definitions import SAVE_PATH_FIGURES

# load data
suffix = ''
window_length = 1024
data = ut.load_data_analysis('coh_icoh_allsubjects_w{}_{}.p'.format(window_length, suffix))
# now make a plot of all selected channels and the histogram over peaks
f = data['frequs']
coh_mat = data['coh_mat']
icoh_mat = data['icoh_mat']

# plot mean
mask = ut.get_array_mask(f > 2, f < 30)
# select the channel with the largest coh
mean = coh_mat.mean(axis=0)
max_idx = np.argmax(np.max(mean, axis=1))
mean = mean[max_idx, :][mask]
se = ut.standard_error(coh_mat[:, max_idx, :], axis=0)[mask]

imean = icoh_mat.mean(axis=0)
max_idx = np.argmax(np.max(imean, axis=1))
imean = imean[max_idx, :][mask]
ise = ut.standard_error(icoh_mat[:, max_idx, :], axis=0)[mask]

plt.figure(figsize=(7, 5))
plt.plot(f[mask], mean, label='mean COH')
plt.fill_between(f[mask], mean + se, mean - se, alpha=.2, color='C0', label='+-standard error')
plt.plot(f[mask], imean, label='mean iCOH')
plt.fill_between(f[mask], imean + ise, imean - ise, alpha=.2, color='C1', label='+-standard error')

plt.title('Interhemispheric coherence and imaginary coherency')
plt.xlabel('Frequency [Hz]')
plt.ylabel('coherence')
plt.legend()
plt.savefig(os.path.join(SAVE_PATH_FIGURES, 'coherence', 'figure4A_{}.pdf'.format(suffix)))
# plt.show()
plt.close()
