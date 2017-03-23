import os

import matplotlib.pyplot as plt
import numpy as np

import utils as ut
from definitions import DATA_PATH

file_list = os.listdir(DATA_PATH)
suffix = '_linear_search_and_amp'
# prelocate arrays for psds and peak amplitudes
window_length = 1024
n_frequ_samples = int(window_length / 2 + 1)
n_patients = 27
n_electrodes = 153
max_amp_psds_theta = np.zeros((n_patients, n_frequ_samples))
max_amp_peak_theta = np.zeros(n_patients)
max_amp_psds_beta = np.zeros((n_patients, n_frequ_samples))
max_amp_peak_beta = np.zeros(n_patients)

theta_peaks = np.zeros(n_electrodes)
beta_peaks = np.zeros(n_electrodes)  # there are supposed to be 153 channels
frequs = None
# subject counters
i_subjects = 0
i_channels = 0

for f in file_list:
    if f.startswith('spmeeg_'):
        d = ut.load_data_spm(f)
        fs = d['fsample'][0][0]
        chanlabels = d['chanlabels'][0]

        # prelocate amp arrays for saving the peak amp of every channel
        beta_amp = np.zeros(d['data'].shape[0])
        theta_amp = np.zeros(d['data'].shape[0])
        psds = np.zeros((d['data'].shape[0], n_frequ_samples))

        plt.figure(figsize=(10, 5))
        for i, lfp in enumerate(d['data']):
            # remove 50Hz noise
            lfp_clean = lfp # band_pass_filter(lfp, band=2, fs=fs, btype='highpass')
            # lfp_clean = remove_50_noise(lfp, fs)
            # calculate psd
            frequs, psd = ut.calculate_psd(lfp_clean, fs=fs, window_length=window_length)

            # normalize like in the paper: take sd of the psd between 4-45 and 55-95Hz
            mask = np.logical_or(ut.get_array_mask(frequs > 5, frequs < 45), ut.get_array_mask(frequs > 55, frequs < 95))
            psd /= np.std(psd[mask])

            # remove 1 / f component
            # frequs_linear, psd_linear = ut.remove_1f_component(frequs, psd)

            # save for later
            psds[i, ] = psd
            # find peak in theta: use the linearized spectra for finding peaks
            idx_theta, peak_amp_theta, ftheta, psd_theta = ut.find_peak_in_band(frequs, psd, [4, 12], linearize=True)
            # take the mean over peak +- 1.5Hz
            # surround_mask = ut.get_array_mask(frequs > (ftheta[idx_theta] - 1.5), frequs < (ftheta[idx_theta] + 1.5))
            theta_amp[i] = peak_amp_theta  # np.mean(psd[surround_mask])
            theta_peaks[i_channels] = ftheta[idx_theta]
            theta_peak = psd_theta[idx_theta]

            # find peak in beta
            idx_beta, peak_amp_beta, fbeta, psd_beta = ut.find_peak_in_band(frequs, psd, [12, 30], linearize=True)
            surround_mask = ut.get_array_mask(frequs > (fbeta[idx_beta] - 1.5), frequs < (fbeta[idx_beta] + 1.5))
            beta_peak = psd_beta[idx_beta]
            beta_amp[i] = peak_amp_beta  # np.mean(psd[surround_mask])
            beta_peaks[i_channels] = fbeta[idx_beta]

            # plotting
            mask = ut.get_array_mask(frequs > 4, frequs < 30)
            plt.subplot(3, 2, i + 1)
            plt.plot(frequs[mask], psd[mask], label='psd')
            plt.plot(ftheta[idx_theta], theta_peak, 'o', label='theta peak')
            plt.plot(fbeta[idx_beta], beta_peak, 'o', label='beta peak')
            plt.legend()
            plt.title('LFP channel {}'.format(chanlabels[i]))
            if not(i == 4 or i == 5):
                plt.xticks([], [])

            # count channels in total
            i_channels += 1

        # now select the channel with highest amplitude
        beta_channel = np.argmax(beta_amp)
        theta_channel = np.argmax(theta_amp)
        max_amp_psds_theta[i_subjects, ] = psds[theta_channel]
        max_amp_peak_theta[i_subjects] = ftheta[theta_channel]

        max_amp_psds_beta[i_subjects, ] = psds[beta_channel]
        max_amp_peak_beta[i_subjects] = fbeta[beta_channel]

        # note which channel was selected
        plt.subplot(3, 2, theta_channel + 1)
        plt.title('LFP channel {}, selected theta'.format(chanlabels[i]))
        plt.subplot(3, 2, beta_channel + 1)
        plt.title('LFP channel {}, selected beta'.format(chanlabels[i]))

        # save figure
        # plt.savefig(os.path.join(SAVE_PATH_FIGURES, f[:-4] + '_psd{}.pdf'.format(suffix)))
        # plt.show()
        plt.close()
        i_subjects += 1

save_dict = dict(filelist=file_list, psd_theta=max_amp_psds_theta, psd_beta=max_amp_psds_beta,
                 theta_peaks_all=theta_peaks, theta_peaks_max=max_amp_peak_theta,
                 beta_peaks_all=beta_peaks, beta_peaks_max=max_amp_peak_beta,
                 frequs=frequs, n_subjects=i_subjects, n_channels=i_channels)
ut.save_data(save_dict, 'psd_maxamp_theta_beta{}.p'.format(suffix))
