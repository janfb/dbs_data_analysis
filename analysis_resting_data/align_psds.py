import os

import matplotlib.pyplot as plt
import numpy as np

import utils as ut
from definitions import SAVE_PATH_FIGURES

"""
Load the data containing the normalized psds that were selected from every subject based on peak amplitude over channels

For the theta and the beta range separately align the peaks and plot for. plot the mean and standard error over subjects
"""


# load data
suffix = '_linear_search_and_amp'
data = ut.load_data_analysis('psd_maxamp_theta_beta{}.p'.format(suffix))
# now make a plot of all selected channels and the histogram over peaks
frequs = data['frequs']
print(data.keys())

n_subjects, n_psd_samples = data['psd_beta'].shape

# exclude subjects with too small peak amps in theta
threshold = 2
subject_mask = data['theta_peaks_max'] > threshold
theta_peaks = data['theta_peaks_max'][subject_mask]
theta_psd = data['psd_theta'][subject_mask]

# for every subject build a mask -3, +8 Hz around the peak,
# collect the psd of this mask in a matrix over subjects
# prelocate with mask around the mean peak
mean_peak_theta = data['theta_peaks_all'].mean()
mean_peak_beta = data['beta_peaks_all'].mean()

frequ_samples = 12

psd_range_mat_theta = -1 * np.ones((subject_mask.sum(), frequ_samples))
psd_range_mat_beta = -1 * np.ones((n_subjects, frequ_samples))

for sub in range(n_subjects):
    # get the peak
    peak_beta = data['beta_peaks_max'][sub]

    # build a mask -3, +8 Hz around the peak
    mask_beta = ut.get_array_mask(frequs > (peak_beta - 3), frequs < (peak_beta + 8))

    beta_frequs = frequs[mask_beta]

    # collect the psd of this mask in a matrix
    psd_range_mat_beta[sub, ] = data['psd_beta'][sub, mask_beta]

for sub in range(np.sum(subject_mask)):
    # get the peak
    peak_theta = theta_peaks[sub]

    # build a mask -3, +8 Hz around the peak
    mask_theta = ut.get_array_mask(frequs > peak_theta - 3, frequs < peak_theta + 8)

    # collect the psd of this mask in a matrix
    psd_range_mat_theta[sub, ] = data['psd_theta'][sub, mask_theta]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
freq_range = np.linspace(-3, 8, frequ_samples)
# comment in to plot individual psds
plt.plot(freq_range, psd_range_mat_theta.T, alpha=.2, color='C1')

# plot standard error around mean
mean = psd_range_mat_theta.mean(axis=0)
se = ut.standard_error(psd_range_mat_theta, axis=0)

plt.plot(freq_range, mean, color='C3', label='mean')
plt.fill_between(freq_range, mean + se, mean - se, alpha=.2, color='C3', label='+-standard error')

plt.xlabel('Frequency [Hz]')
plt.ylabel('Relative spectral power [au]')
plt.title('Theta peaks above {}Hz aligned'.format(threshold))
plt.ylim([0, 7])
plt.legend()

plt.subplot(1, 2, 2)
freq_range = np.linspace(-3, 8, frequ_samples)
# comment in to plot individual psds
plt.plot(freq_range, psd_range_mat_beta.T, alpha=.2, color='C1')

# plot standard error around mean
mean = psd_range_mat_beta.mean(axis=0)
se = ut.standard_error(psd_range_mat_beta, axis=0)

plt.plot(freq_range, mean, color='C2', label='mean')
plt.fill_between(freq_range, mean + se, mean - se, alpha=.2, color='C2', label='+-standard error')

plt.xlabel('Frequency [Hz]')
plt.ylabel('Relative spectral power [au]')
plt.title('Beta peaks aligned')
plt.ylim([0, 7])
plt.legend()

plt.savefig(os.path.join(SAVE_PATH_FIGURES, 'figure_3AB_{}Hz_cutoff_all.pdf'.format(threshold)))
# plt.show()
plt.close()