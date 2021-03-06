import os
import numpy as np
import utils as ut
from definitions import SAVE_PATH_DATA, DATA_PATH, SAVE_PATH_FIGURES
import matplotlib.pyplot as plt
import scipy.signal

"""
read in the selected subject data

look at raw and psd to check for artifacts

band-pass filter in theta

calculate instantaneous phase

save to new dicts
"""

data_folder = os.path.join(DATA_PATH, 'good_theta')
save_folder = os.path.join(SAVE_PATH_DATA, 'phase')

# read all files in the data folder
file_list = [f for f in os.listdir(data_folder) if f.startswith('spmeeg')]

frequ_range = 'theta'
if frequ_range == 'theta':
    band = np.array([4., 12.])
elif frequ_range == 'beta':
    band = np.array([12., 30.])
else:
    band = None

# amount of time to be considered
seconds = 5
seconds_str = '{}'.format(seconds)

# for every subject file
for sub, sub_file in enumerate(file_list):
    subject_number = int(sub_file[7:-4])
    if subject_number > 0:
        # load data
        d = ut.load_data_spm(sub_file, data_folder=data_folder)
        fs = d['fsample'][0][0]
        channels = d['chanlabels'][0]
        print('analysing subject file {}'.format(sub_file))

        # make new dict
        subject_dict = dict(lfp_band={}, channels=channels, phase={}, fs=fs)

        plt.figure(figsize=(10, 7))

        for chan_idx, chan in enumerate(channels):
            # mean center
            lfp = d['data'][chan_idx] - d['data'][chan_idx].mean()

            # filter around the theta peak
            lfp_band = ut.band_pass_filter(lfp, fs, band=band, plot_response=False)

            # extract instantaneous phase
            analystic_signal = scipy.signal.hilbert(lfp_band)
            phase = np.arctan2(np.imag(analystic_signal), np.real(analystic_signal)) + np.pi

            # save to dict
            subject_dict['phase'][chan[0]] = phase
            subject_dict['lfp_band'][chan[0]] = lfp_band

            # use only part of the data
            start = 4000
            stop = start + seconds * fs
            # seconds is -1 if the whole data shall be used
            if seconds == -1:
                stop = phase.size
                seconds_str = 'all'
            # plot
            hist, bins = np.histogram(phase[start:stop], bins=60)
            radii = hist / phase[start:stop].size
            theta = 0.5 * (bins[:-1] + bins[1:])
            ax = plt.subplot(2, 3, chan_idx + 1, projection='polar')
            bars = ax.bar(theta, radii)
            ax.set_rticks(np.round(np.linspace(0, np.max(radii), 3), 4))
            plt.title(chan[0])
        # start = 1000
        # stop = start + 5000
        # plt.figure(figsize=(10, 7))
        # plt.subplot(2, 1, 1)
        # plt.plot(lfp[start:stop])
        # plt.plot(lfp_band[start:stop])
        # # plt.plot(lfp_broad[start:stop])
        #
        # # plot phase
        # plt.subplot(2, 1, 2)
        # plt.plot(phase[start:stop])
        # plt.show()

        plt.suptitle('Instantaneous phase distribution, {} seconds of lfp data'.format(seconds_str))
        # plt.show()
        filename_figure = '{}_subject_{}_phase_histogram_{}sec.pdf'.format(frequ_range, subject_number, seconds_str)
        plt.savefig(os.path.join(SAVE_PATH_FIGURES, 'phase', filename_figure))
        plt.close()
