import os
import numpy as np
import utils as ut
from definitions import SAVE_PATH_DATA_BAROW, DATA_PATH_BAROW, SAVE_PATH_FIGURES_BAROW
import matplotlib.pyplot as plt
import scipy.signal

"""
read in the selected subject data

look at raw and psd to check for artifacts

band-pass filter in theta

calculate instantaneous phase

save to new dicts
"""

data_folder = os.path.join(SAVE_PATH_DATA_BAROW, 'cleaned')
save_folder = os.path.join(SAVE_PATH_DATA_BAROW, 'phase')

# read all files in the data folder
file_list = [f for f in os.listdir(data_folder) if f.startswith('subject')]

frequ_range = 'theta'
if frequ_range == 'theta':
    band = np.array([4., 12.])
elif frequ_range == 'beta':
    band = np.array([12., 30.])
else:
    band = None

# use only a certain time range of the data
seconds = 2
seconds_str = '{}'.format(seconds)

# for every subject file
for sub, sub_file in enumerate(file_list):
    # load data
    d = ut.load_data_analysis(sub_file, data_folder=data_folder)
    print('analysing subject file {}'.format(sub_file))
    subject_number = d['number']
    conditions = d['conditions']
    condition_order = ['rest', 'stim', 'poststim']

    # make new dict
    subject_dict = dict(lfp_band={}, conditions=conditions, phase={})

    plt.figure(figsize=(10, 5))

    for cond_idx, condition in enumerate(conditions):
        # mean center
        lfp = d['lfp'][condition] - d['lfp'][condition].mean()
        fs = d['fs'][condition]

        # filter around the theta peak
        lfp_band = ut.band_pass_filter(lfp, fs, band=band, plot_response=False)

        # extract instantaneous phase
        # take only a part of the data to save time
        start = 5 * fs  # 5 sec offset
        stop = start + 240 * fs  # take 4 minutes of recording maximum
        if stop > lfp_band.size:
            stop = -1
        analystic_signal = scipy.signal.hilbert(lfp_band[start:stop])
        phase = np.arctan2(np.imag(analystic_signal), np.real(analystic_signal)) + np.pi

        # save to dict
        subject_dict['phase'][condition] = phase
        subject_dict['lfp_band'][condition] = lfp_band
        subject_dict['fs'] = fs

        # use only part of the data
        start = 4000
        stop = start + seconds * fs
        # seconds is -1 if the whole data shall be used
        if seconds == -1:
            stop = phase.size
            seconds_str = 'all'
        print('Using {} seconds of lfp data'.format(seconds_str))

        # plot
        hist, bins = np.histogram(phase[start:stop], bins=60)
        radii = hist / phase.size
        theta = 0.5 * (bins[:-1] + bins[1:])
        ax = plt.subplot(1, 3, cond_idx + 1, projection='polar')
        bars = ax.bar(theta, radii)
        ax.set_rticks(np.round(np.linspace(0, np.max(radii), 3), 4))
        plt.title(condition)

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
    filename_figure = '{}_subject_{}_phase_histogram_{}.pdf'.format(frequ_range, subject_number, seconds_str)
    plt.savefig(os.path.join(SAVE_PATH_FIGURES_BAROW, 'phase', filename_figure))
    plt.close()