import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from definitions import DATA_PATH, SAVE_PATH_FIGURES

file_list = os.listdir(DATA_PATH)
suffix = '_linear_search_and_amp'
# prelocate arrays for psds and peak amplitudes
window_length = 1024
n_frequ_samples = int(window_length / 2 + 1)
n_patients = 27
n_electrodes = 153
max_amp_psds_theta = np.zeros((n_patients, n_frequ_samples))
max_amp_peak_frequency_theta = np.zeros(n_patients)
max_amp_psds_beta = np.zeros((n_patients, n_frequ_samples))
max_amp_peak_frequency_beta = np.zeros(n_patients)

theta_peaks = np.zeros(n_electrodes)
beta_peaks = np.zeros(n_electrodes)  # there are supposed to be 153 channels
frequs = None
# subject counters
subject_idx = 0
electrode_idx = 0

for f in file_list:
    if f.startswith('spmeeg_'):
        d = ut.load_data_spm(f)
        fs = d['fsample'][0][0]
        chanlabels = d['chanlabels'][0]

        # prelocate amp arrays for saving the peak amp of every channel
        n_channels = d['data'].shape[0]
        beta_amp = np.zeros(n_channels)
        theta_amp = np.zeros(n_channels)
        psds = np.zeros((n_channels, n_frequ_samples))
        theta_peak_frequency = np.zeros(n_channels)
        beta_peak_frequency = np.zeros(n_channels)

        plt.figure(figsize=(10, 5))
        for channel_idx, lfp in enumerate(d['data']):
            # remove 50Hz noise
            lfp_clean = lfp
            # lfp_clean = ut.band_stop_filter(lfp, fs, band=[49, 51], plot_response=False)
            # calculate psd
            frequs, psd = ut.calculate_psd(lfp_clean, fs=fs, window_length=window_length)

            # normalize like in the paper: take sd of the psd between 4-45 and 55-95Hz
            mask = np.logical_or(ut.get_array_mask(frequs > 5, frequs < 45), ut.get_array_mask(frequs > 55, frequs < 95))
            psd /= np.std(psd[mask])

            # remove 1 / f component
            # frequs_linear, psd_linear = ut.remove_1f_component(frequs, psd)

            # save for later
            psds[channel_idx,] = psd
            # find peak in theta: use the linearized spectra for finding peaks
            idx_theta, peak_amp_theta, ftheta, psd_theta = ut.find_peak_in_band(frequs, psd, [4, 12], linearize=True)
            # save the frequency at which the peak occured, in one long list for the histogram
            theta_peaks[electrode_idx] = ftheta[idx_theta]
            # save it in a matrix to select it per subject for the alignment
            theta_peak_frequency[channel_idx] = ftheta[idx_theta]
            # save the amplitude of the normalized psd
            theta_peak = psd_theta[idx_theta]
            # save the amplitude of the theta peak from the search algorithm
            theta_amp[channel_idx] = peak_amp_theta  # np.mean(psd[surround_mask])

            # find peak in beta
            idx_beta, peak_amp_beta, fbeta, psd_beta = ut.find_peak_in_band(frequs, psd, [12, 30], linearize=True)
            surround_mask = ut.get_array_mask(frequs > (fbeta[idx_beta] - 1.5), frequs < (fbeta[idx_beta] + 1.5))
            beta_peak = psd_beta[idx_beta]
            beta_amp[channel_idx] = peak_amp_beta  # np.mean(psd[surround_mask])
            beta_peak_frequency[channel_idx] = fbeta[idx_beta]
            beta_peaks[electrode_idx] = fbeta[idx_beta]

            # plotting
            mask = ut.get_array_mask(frequs > 4, frequs < 30)
            plt.subplot(3, 2, channel_idx + 1)
            plt.plot(frequs[mask], psd[mask], label='psd')
            plt.plot(ftheta[idx_theta], theta_peak, 'o', label='theta peak')
            plt.plot(fbeta[idx_beta], beta_peak, 'o', label='beta peak')
            plt.legend()
            plt.title('LFP channel {}'.format(chanlabels[channel_idx]))
            if not(channel_idx == 4 or channel_idx == 5):
                plt.xticks([], [])

            # count channels in total
            electrode_idx += 1

        # now select the channel with highest amplitude
        beta_channel = np.argmax(beta_amp)
        theta_channel = np.argmax(theta_amp)
        max_amp_psds_theta[subject_idx,] = psds[theta_channel,]
        max_amp_peak_frequency_theta[subject_idx] = theta_peak_frequency[theta_channel]

        max_amp_psds_beta[subject_idx,] = psds[beta_channel]
        max_amp_peak_frequency_beta[subject_idx] = beta_peak_frequency[beta_channel]

        # note which channel was selected
        plt.subplot(3, 2, theta_channel + 1)
        plt.axvline(max_amp_peak_frequency_theta[subject_idx])
        plt.title('LFP channel {}, selected theta'.format(chanlabels[channel_idx]))
        plt.subplot(3, 2, beta_channel + 1)
        plt.axvline(max_amp_peak_frequency_beta[subject_idx])
        plt.title('LFP channel {}, selected beta'.format(chanlabels[channel_idx]))

        # save figure
        plt.savefig(os.path.join(SAVE_PATH_FIGURES, f[:-4] + '_psd{}.pdf'.format(suffix)))
        # plt.show()
        plt.close()
        subject_idx += 1

save_dict = dict(filelist=file_list, psd_theta=max_amp_psds_theta, psd_beta=max_amp_psds_beta,
                 theta_peaks_all=theta_peaks, theta_peaks_max=max_amp_peak_frequency_theta,
                 beta_peaks_all=beta_peaks, beta_peaks_max=max_amp_peak_frequency_beta,
                 frequs=frequs, n_subjects=subject_idx, n_channels=electrode_idx)
ut.save_data(save_dict, 'psd_maxamp_theta_beta{}.p'.format(suffix))
