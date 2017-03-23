import os

import matplotlib.pyplot as plt
import numpy as np

from definitions import SAVE_PATH_FIGURES, DATA_PATH
from utils import load_data_spm, band_pass_filter

file_list = os.listdir(DATA_PATH)

i_subjects = 0
i_channels = 0

for f in file_list:
    if f.startswith('spmeeg_'):
        d = load_data_spm(f)
        fs = d['fsample'][0][0]
        dt = 1 / fs
        t = np.arange(0, d['data'].shape[1] * dt, dt)
        chanlabels = d['chanlabels'][0]

        plt.figure(figsize=(10, 5))
        for i, lfp in enumerate(d['data']):
            # remove 50Hz noise or band pass filter
            band = [1, 45]
            lfp_clean = band_pass_filter(lfp, band=band, fs=fs)
            # lfp_clean = remove_50_noise(lfp, fs)
            plt.subplot(3, 2, i + 1)
            plt.plot(t[::200], lfp_clean[0::200], linewidth=.7)
            plt.legend()
            plt.title('LFP channel {}'.format(chanlabels[i]))
            if not (i == 4 or i == 5):
                plt.xticks([], [])
            else:
                plt.xlabel('time [s]')
            if not i % 2:
                plt.ylabel('lfp [$\mu$ V]')

            # count channels in total
            i_channels += 1

        # save figure
        plt.savefig(os.path.join(SAVE_PATH_FIGURES, 'lfp_band', f[:-4] + '_lfp_band{}_{}.pdf'.format(band[0], band[1])))
        # plt.show()
        plt.close()
        i_subjects += 1
