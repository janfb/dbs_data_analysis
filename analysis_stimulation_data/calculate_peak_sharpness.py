import os
import numpy as np
import utils as ut
from definitions import SAVE_PATH_DATA_BAROW, SAVE_PATH_FIGURES_BAROW
import scipy.signal
import matplotlib.pyplot as plt

"""
read filtered data and calculate statistics like in cole paper, save the results in subject specific dictionaries for
later analysis
"""

data_folder = os.path.join(SAVE_PATH_DATA_BAROW, 'cleaned')
save_folder = os.path.join(SAVE_PATH_DATA_BAROW, 'analysis')

# read all files in the data folder
file_list = [f for f in os.listdir(data_folder) if f.startswith('subject')]
lfp_cutoff = 1
n_conditions = 3
condition_order = ['rest', 'stim', 'poststim']
esr_matrix = np.zeros((len(file_list), n_conditions))

# band pass filter in theta (or beta?)
frequ_range = 'theta'
if frequ_range == 'theta':
    band = np.array([3., 13.])
elif frequ_range == 'beta':
    band = np.array([13., 30.])
else:
    band = None

# for every subject file
for sub, sub_file in enumerate(file_list):
    # load data
    d = ut.load_data_analysis(sub_file, data_folder=data_folder)
    print('analysing subject file {}'.format(sub_file))
    # make new entries in the dict
    d['sharpness'] = {}
    d['steepness'] = {}

    # for all three conditions
    for i in range(len(d['conditions'])):
        # select the condition in the defined order
        c = d['conditions'][condition_order[i]]
        print('Condition {}'.format(c))
        d['sharpness'][c] = {}
        d['steepness'][c] = {}

        # get the data
        lfp_raw = d['lfp'][c]
        fs = d['fs'][c]

        # preprocess
        # remove power noise
        nyq = fs / 2
        wn = np.array([48, 52]) / nyq
        # noinspection PyTupleAssignmentBalance
        b, a = scipy.signal.butter(3, wn, btype='bandstop')
        lfp_pre = scipy.signal.lfilter(b, a, lfp_raw)

        # low pass filter at 200hz, FIR, window method, numtaps = 250ms = 250 samples
        cut_off_normalized = 200 / nyq
        coefs = scipy.signal.firwin(numtaps=250, cutoff=cut_off_normalized)
        lfp_pre = scipy.signal.filtfilt(coefs, 1., lfp_pre)
        lfp_pre -= lfp_pre.mean()

        # filter

        wn = np.array(band) / fs * 2
        # noinspection PyTupleAssignmentBalance
        b, a = scipy.signal.butter(2, wn, btype='bandpass')
        lfp_band = scipy.signal.filtfilt(b, a, lfp_pre)
        # lfp_band = ut.band_pass_filter(lfp_raw, fs, band=band, plot_response=False)
        # cut the beginning and the end of the time series to avoid artifacts
        lfp_band = lfp_band[lfp_cutoff * fs: -lfp_cutoff * fs]
        lfp_band -= lfp_band.mean()
        lfp_pre = lfp_pre[lfp_cutoff * fs: -lfp_cutoff * fs]

        # look at the spectogram to select time periods of higher beta
        t, burst_mask = ut.select_time_periods_of_high_power(data=lfp_pre, band=band)
        # burst_mask = np.logical_not(np.zeros_like(lfp_pre))
        # plt.plot(t, lfp_pre)
        # plt.fill_between(t, np.min(lfp_pre), np.max(lfp_pre), where=burst_mask, alpha=.4, color='red')
        # plt.show()

        # find rising and falling zero crossings using the filtered data

        # calculate zero crossings on band data
        zeros_rising, zeros_falling, zeros = ut.find_rising_and_falling_zeros(lfp_band[burst_mask])

        # find the peaks in between the zeros: use the RAW DATA for this step.
        analysis_lfp = lfp_pre[burst_mask]
        peaks, troughs, extrema = ut.find_peaks_and_troughs(analysis_lfp, zeros)

        # calculate peak sharpness:
        peak_sharpness = ut.calculate_peak_sharpness(analysis_lfp, peaks, fs=fs)
        trough_sharpness = ut.calculate_peak_sharpness(analysis_lfp, troughs, fs=fs)
        mean_peak_sharpness = np.mean(peak_sharpness)
        mean_trough_sharpness = np.mean(trough_sharpness)
        # extrema sharpness ratio, from the paper
        esr = np.max([mean_peak_sharpness / mean_trough_sharpness, mean_trough_sharpness / mean_peak_sharpness])
        esr_matrix[sub, i] = esr

        # calculate the steepness
        rise_steepness, fall_steepness, steepness_indices= ut.calculate_rise_and_fall_steepness(analysis_lfp, extrema)
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

        # # check the zero crossing issue
        # plt.plot(lfp_for_sharpness_anaylsis)
        # plt.plot(lfp_band)
        # plt.plot(zeros_falling, np.zeros_like(zeros_falling), 'ro')
        # plt.plot(zeros_rising, np.zeros_like(zeros_rising), 'bo')
        # plt.plot(peaks, lfp_for_sharpness_anaylsis[peaks], 'yo')
        # plt.plot(troughs, lfp_for_sharpness_anaylsis[troughs], 'co')
        # plt.show()

        # break

    # save data
    # remove large lfp array to save space
    d['lfp'] = None
    ut.save_data(data_dict=d,
                 filename='subject_{}_sharpness_steepness_{}.p'.format(d['number'], frequ_range),
                 folder=save_folder)

    # do it for one subject only
    # break

# plot overview over mean phase vector
plt.figure(figsize=(8, 5))
plt.title('Extrema sharpness ration over conditions')
plt.plot(esr_matrix.mean(axis=0), 'o-', linewidth=2)
plt.plot(esr_matrix.T, alpha=.3)
plt.legend(np.hstack((['mean'], np.arange(1, 17))), loc=0, prop={'size': 7})
plt.xticks(np.arange(3), condition_order)
plt.ylabel('esr')
filename_figure = '{}_esr_over_subjects_bursts.pdf'.format(frequ_range)
plt.savefig(os.path.join(SAVE_PATH_FIGURES_BAROW, 'sharpness', filename_figure))
plt.show()
plt.close()
