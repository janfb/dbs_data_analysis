import os
import numpy as np
import utils as ut
from definitions import SAVE_PATH_DATA_BAROW

"""
read filtered data and calculate statistics like in cole paper, save the results in subject specific dictionaries for
later analysis
"""

data_folder = os.path.join(SAVE_PATH_DATA_BAROW, 'cleaned')
save_folder = os.path.join(SAVE_PATH_DATA_BAROW, 'analysis')

# read all files in the data folder
file_list = [f for f in os.listdir(data_folder) if f.startswith('subject')]

# for every subject file
for sub, sub_file in enumerate(file_list):
    # load data
    d = ut.load_data_analysis(sub_file, data_folder=data_folder)
    print('analysing subject file {}'.format(sub_file))
    # make new entries in the dict
    d['sharpness'] = {}
    d['steepness'] = {}

    # for all three conditions
    for i, c in enumerate(d['conditions']):
        print('Condition {}'.format(c))
        d['sharpness'][c] = {}
        d['steepness'][c] = {}

        # zero mean the data
        data = d['lfp'][c] - np.mean(d['lfp'][c])
        # data = data[:1000]

        # band pass filter in theta (or beta?)
        frequ_range = 'theta'
        if frequ_range == 'theta':
            band = np.array([3., 13.])
        elif frequ_range == 'beta':
            band = np.array([13., 30.])
        else:
            band = None

        fs = d['fs'][c]
        lfp_band = ut.band_pass_filter(data, fs, band=band, plot_response=False)

        # find rising and falling zero crossings
        zeros_rising, zeros_falling, zeros = ut.find_rising_and_falling_zeros(lfp_band)

        # find the peaks in between the zeros
        peaks, troughs, extrema = ut.find_peaks_and_troughs(lfp_band, zeros)

        # calculate peak sharpness:
        peak_sharpness = ut.calculate_peak_sharpness(lfp_band, peaks, fs=fs)
        trough_sharpness = ut.calculate_peak_sharpness(lfp_band, troughs, fs=fs)
        mean_peak_sharpness = np.mean(peak_sharpness)
        mean_trough_sharpness = np.mean(trough_sharpness)
        # extrema sharpness ratio, from the paper
        esr = np.max([mean_peak_sharpness / mean_trough_sharpness, mean_trough_sharpness / mean_peak_sharpness])

        # calculate the steepness
        rise_steepness, fall_steepness = ut.calculate_rise_and_fall_steepness(lfp_band, extrema)
        mean_rise_steepness = np.mean(rise_steepness)
        mean_fall_steepness = np.mean(fall_steepness)
        # rise decay steepness ratio
        rdsr = np.max([mean_rise_steepness / mean_fall_steepness, mean_fall_steepness / mean_rise_steepness])

        # save to dict
        d['sharpness'][c]['trough_sharpness'] = trough_sharpness
        d['sharpness'][c]['peak_sharpness'] = peak_sharpness
        d['sharpness'][c]['trough_average'] = mean_trough_sharpness
        d['sharpness'][c]['peak_average'] = mean_peak_sharpness
        d['sharpness'][c]['esr'] = esr

        d['steepness'][c]['rise'] = rise_steepness
        d['steepness'][c]['fall'] = fall_steepness
        d['steepness'][c]['rise_average'] = mean_rise_steepness
        d['steepness'][c]['fall_average'] = mean_fall_steepness
        d['steepness'][c]['rdsr'] = rdsr

        d['frequ_range'] = frequ_range


        # plt.plot(data)
        # plt.plot(lfp_band)
        # plt.plot(zeros_falling, np.zeros_like(zeros_falling), 'ro')
        # plt.plot(zeros_rising, np.zeros_like(zeros_rising), 'bo')
        # plt.plot(peaks, lfp_band[peaks], 'yo')
        # plt.plot(troughs, lfp_band[troughs], 'co')
        # plt.show()

        # break

    # for i, c in enumerate(d['conditions']):
    #     plt.subplot(1, 3, i + 1)
    #     plt.title(c)
    #     plt.hist(d['sharpness'][c]['trough_sharpness'], label='troughs', alpha=.5)
    #     plt.hist(d['sharpness'][c]['peak_sharpness'], label='peaks', alpha=.5)
    #
    # plt.legend()
    # plt.show()

    # save data
    # remove large lfp array to save space
    d['lfp'] = None
    ut.save_data(data_dict=d,
                 filename='subject_{}_sharpness_steepness_{}.p'.format(d['number'], frequ_range),
                 folder=save_folder)

    # do it for one subject only
    # break
