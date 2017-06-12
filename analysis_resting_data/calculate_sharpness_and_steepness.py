import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from definitions import DATA_PATH, SAVE_PATH_DATA

"""
Calculate the peak sharpness and rise and fall steepness for the resting dystonia patients
"""

data_folder = DATA_PATH
save_folder = os.path.join(SAVE_PATH_DATA, 'sharpness_steepness')


# band pass filter in theta (or beta?)
frequ_range = 'theta'
if frequ_range == 'theta':
    band = np.array([3., 13.])
elif frequ_range == 'beta':
    band = np.array([13., 30.])
else:
    band = None

file_list = os.listdir(DATA_PATH)

n_patients = 27
n_electrodes = 153

# subject counters
i_subjects = 0
i_channels = 0

for file in file_list:
    if file.startswith('spmeeg_'):
        print('Processing file: {}'.format(file))
        d = ut.load_data_spm(file)
        fs = d['fsample'][0][0]
        channels = d['chanlabels'][0]

        # make new dict
        subject_dict = dict(channels=channels, phase={}, fs=fs,
                            sharpness={},
                            steepness={},
                            subject_number=int(file[7:-4]))

        # for every lfp channel
        for i, lfp in enumerate(d['data']):
            current_channel = channels[i][0]

            # prelocat the dictionary for all channels
            print('Channel {}'.format(current_channel))
            subject_dict['sharpness'][current_channel] = {}
            subject_dict['steepness'][current_channel] = {}

            # zero mean the lfp signal
            data = lfp - np.mean(lfp)

            # band pass filter
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
            rise_steepness, fall_steepness, steepness_indices = ut.calculate_rise_and_fall_steepness(lfp_band, extrema)
            mean_rise_steepness = np.mean(rise_steepness)
            mean_fall_steepness = np.mean(fall_steepness)
            # rise decay steepness ratio
            rdsr = np.max([mean_rise_steepness / mean_fall_steepness, mean_fall_steepness / mean_rise_steepness])

            # save to dict
            subject_dict['sharpness'][current_channel]['trough_sharpness'] = trough_sharpness
            subject_dict['sharpness'][current_channel]['peak_sharpness'] = peak_sharpness
            subject_dict['sharpness'][current_channel]['trough_average'] = mean_trough_sharpness
            subject_dict['sharpness'][current_channel]['peak_average'] = mean_peak_sharpness
            subject_dict['sharpness'][current_channel]['esr'] = esr

            subject_dict['steepness'][current_channel]['rise'] = rise_steepness
            subject_dict['steepness'][current_channel]['fall'] = fall_steepness
            subject_dict['steepness'][current_channel]['rise_average'] = mean_rise_steepness
            subject_dict['steepness'][current_channel]['fall_average'] = mean_fall_steepness
            subject_dict['steepness'][current_channel]['rdsr'] = rdsr

            subject_dict['frequ_range'] = frequ_range

        # save figure
        ut.save_data(data_dict=subject_dict,
                     filename='subject_{}_sharpness_steepness_{}.p'.format(i_subjects + 1, frequ_range),
                     folder=save_folder)
        i_subjects += 1




