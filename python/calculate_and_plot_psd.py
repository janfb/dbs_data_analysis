import os
import numpy as np
import matplotlib.pyplot as plt
from definitions import SAVE_PATH_FIGURES, SAVE_PATH_DATA, DATA_PATH
from python.utils import load_data_spm, remove_50_noise, calculate_psd, find_peak_in_band, get_array_mask, save_data, band_pass_filter


file_list = os.listdir(DATA_PATH)

# prelocate arrays for psds and peak amplitudes
n_frequ_samples = 513
n_patients = 27
n_electrodes = 153
max_amp_psds_theta = np.zeros((n_patients, n_frequ_samples))
max_amp_psds_beta = np.zeros((n_patients, n_frequ_samples))

theta_peaks = np.zeros(n_electrodes)
beta_peaks = np.zeros(n_electrodes)  # there are supposed to be 153 channels
frequs = None
# subject counters
i_subjects = 0
i_channels = 0

for f in file_list:
    if f.startswith('spmeeg_'):
        d = load_data_spm(f)
        fs = d['fsample'][0][0]
        chanlabels = d['chanlabels'][0]

        # prelocate amp arrays for saving the peak amp of every channel
        beta_amp = np.zeros(d['data'].shape[0])
        theta_amp = np.zeros(d['data'].shape[0])
        psds = np.zeros((d['data'].shape[0], n_frequ_samples))

        plt.figure(figsize=(10, 5))
        for i, lfp in enumerate(d['data']):
            # remove 50Hz noise
            lfp_clean = lfp  # band_pass_filter(lfp, band=[4, 45], fs=fs)
            # lfp_clean = remove_50_noise(lfp, fs)
            # calculate psd
            frequs, psd = calculate_psd(lfp_clean, fs)
            # normalize like in the paper
            mask = np.logical_or(get_array_mask(frequs > 5, frequs < 45), get_array_mask(frequs > 55, frequs < 95))
            psd /= np.std(psd[mask])
            # save for later
            psds[i, ] = psd
            # find peak in theta
            idx_theta, ftheta, psd_theta = find_peak_in_band(frequs, psd, [4, 12])
            # take the mean over peak +- 1.5Hz
            # ignore theta peaks at the border
            if idx_theta == 0:
                theta_amp[i] = 0
                theta_peaks[i_channels] = ftheta[idx_theta]
            else:
                surround_mask = get_array_mask(ftheta > (ftheta[idx_theta] - 1.5), ftheta < (ftheta[idx_theta] + 1.5))
                theta_amp[i] = np.mean(psd_theta[surround_mask])
                theta_peaks[i_channels] = ftheta[idx_theta]

            # find peak in beta
            idx_beta, fbeta, psd_beta = find_peak_in_band(frequs, psd, [12, 30])
            surround_mask = get_array_mask(fbeta > (fbeta[idx_beta] - 1.5), fbeta < (fbeta[idx_beta] + 1.5))
            beta_amp[i] = np.mean(psd_beta[surround_mask])
            beta_peaks[i_channels] = fbeta[idx_beta]

            # plotting
            mask = get_array_mask(frequs > 4, frequs < 30)
            plt.subplot(3, 2, i + 1)
            plt.plot(frequs[mask], psd[mask], label='psd')
            plt.plot(ftheta[idx_theta], psd_theta[idx_theta], 'o', label='theta peak')
            plt.plot(fbeta[idx_beta], psd_beta[idx_beta], 'o', label='beta peak')
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
        max_amp_psds_beta[i_subjects, ] = psds[beta_channel]

        # note which channel was selected
        plt.subplot(3, 2, theta_channel + 1)
        plt.title('LFP channel {}, selected theta'.format(chanlabels[i]))
        plt.subplot(3, 2, beta_channel + 1)
        plt.title('LFP channel {}, selected beta'.format(chanlabels[i]))

        # save figure
        plt.savefig(os.path.join(SAVE_PATH_FIGURES, f[:-4] + '_psd.pdf'))
        # plt.show()
        plt.close()
        i_subjects += 1

save_dict = dict(filelist=file_list, psd_theta=max_amp_psds_theta, psd_beta=max_amp_psds_beta, theta_peaks=theta_peaks,
                 beta_peaks=beta_peaks, frequs=frequs, n_subjects=i_subjects, n_channels=i_channels)
save_data(save_dict, 'psd_maxamp_theta_beta.p')
