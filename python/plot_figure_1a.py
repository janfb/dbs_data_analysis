import python.utils as ut
import matplotlib.pyplot as plt
import numpy as np
from definitions import SAVE_PATH_FIGURES
import os

# load data
suffix = '_linear_search'
data = ut.load_data_analysis('psd_maxamp_theta_beta{}.p'.format(suffix))
# now make a plot of all selected channels and the histogram over peaks
frequs = data['frequs']

mask = ut.get_array_mask(frequs > 2, frequs < 40)
plt.figure(figsize=(7, 5))
# plot individual stectra
# plt.plot(frequs[mask], data['psd_beta'][:, mask].T, linewidth=.7, color='C1', alpha=.5)
# plt.plot(frequs[mask], data['psd_theta'][:, mask].T, linewidth=.7, color='C4', alpha=.5)

# plot std envelope
joined_psd_mean = np.mean(np.vstack((data['psd_theta'], data['psd_beta'])), axis=0)
se = ut.standard_error(np.vstack((data['psd_theta'], data['psd_beta'])), axis=0)
plt.fill_between(frequs[mask], joined_psd_mean[mask] + se[mask], joined_psd_mean[mask] - se[mask], alpha=.2,
                 color='C3', label='+-standard error')
plt.plot(frequs[mask], joined_psd_mean[mask], label='mean', color='C3')

plt.xlabel('Frequency [Hz]')
plt.ylabel('Relative power [a.u.]')
plt.legend()
plt.title('Average spectra across all patients')
plt.savefig(os.path.join(SAVE_PATH_FIGURES, 'figure_1A{}.pdf'.format(suffix)))
plt.show()

# plot histogram
plt.figure(figsize=(7, 5))
plt.hist(x=data['theta_peaks'], bins=27, range=[3, 30], label='theta')
plt.hist(x=data['beta_peaks'], bins=27, range=[3, 30], label='beta')

plt.title('Peak histogram for theta and beta')
plt.xlabel('Frequency [Hz]')
plt.ylabel('count')
plt.legend()
plt.savefig(os.path.join(SAVE_PATH_FIGURES, 'figure_1B{}.pdf'.format(suffix)))
plt.show()
