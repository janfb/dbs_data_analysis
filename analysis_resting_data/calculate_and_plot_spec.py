import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import utils as ut
from definitions import DATA_PATH, SAVE_PATH_FIGURES

"""
Plot the spectogram as in the supplementary material of Neumann et al 2017 manuscript
"""


file_list = os.listdir(DATA_PATH)

# prelocate arrays for psds and peak amplitudes
window_length = 1024
n_frequ_samples = int(window_length / 2 + 1)
n_patients = 27
n_electrodes = 153

frequs = None
# subject counters
i_subjects = 0
i_channels = 0

for file in file_list:
    if file.startswith('spmeeg_'):
        d = ut.load_data_spm(file)
        fs = d['fsample'][0][0]
        channel_labels = d['chanlabels'][0]

        fig = plt.figure(figsize=(15, 8))
        for i, lfp in enumerate(d['data']):

            f, t, sxx = ut.calculate_spectrogram(lfp, fs=fs)

            # transform to log
            ssx_log = np.log(sxx)

            # define a mask and build a grid
            mask = ut.get_array_mask(f > 4, f < 12)
            xgrid, ygrid = np.meshgrid(f[mask], t)

            # plotting
            ax = fig.add_subplot(3, 2, i + 1, projection='3d')

            surf = ax.plot_surface(X=xgrid, Y=ygrid, Z=ssx_log[mask, :].T, cmap='viridis')
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel('Time [s]')
            ax.set_zlabel('log power ')
            fig.colorbar(surf, shrink=0.5, aspect=5)

        # save figure
        filename_save = file[:-4] + '_spectro.pdf'
        plt.savefig(os.path.join(SAVE_PATH_FIGURES, filename_save))
        # plt.show()
        plt.close()
        i_subjects += 1




