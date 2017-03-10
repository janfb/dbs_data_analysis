import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from definitions import SAVE_PATH_FIGURES, SAVE_PATH_DATA, DATA_PATH
from python.utils import load_data, remove_50_noise, calculate_psd, find_peak_in_band, get_array_mask


file_list = os.listdir(DATA_PATH)

for f in file_list:
    if f.startswith('spmeeg_'):
        d = load_data(f)
        fs = d['fsample'][0][0]
        chanlabels = d['chanlabels'][0]
        plt.figure(figsize=(10, 5))
        for i, lfp in enumerate(d['data']):
            # remove 50Hz noise
            lfp_clean = remove_50_noise(lfp, fs=fs)
            # calculate psd
            frequs, psd = calculate_psd(lfp_clean, fs)
            # find peak in theta
            idx_theta, ftheta, psd_theta = find_peak_in_band(frequs, psd, [4, 12])
            # find peak in theta
            idx_beta, fbeta, psd_beta = find_peak_in_band(frequs, psd, [12, 30])
            mask = get_array_mask(frequs > 4, frequs < 41)
            plt.subplot(3, 2, i + 1)
            plt.plot(frequs[mask], psd[mask], label='psd')
            plt.plot(ftheta[idx_theta], psd_theta[idx_theta], 'o', label='theta peak')
            plt.plot(fbeta[idx_beta], psd_beta[idx_beta], 'o', label='beta peak')
            plt.legend()
            plt.title('LFP channel {}'.format(chanlabels[i]))
            if not(i == 4 or i == 5):
                plt.xticks([], [])

        plt.savefig(os.path.join(SAVE_PATH_FIGURES, f[:-3] + 'pdf'))
        plt.close()
        # plt.show()





