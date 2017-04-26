import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from definitions import SAVE_PATH_FIGURES_BAROW, SAVE_PATH_DATA_BAROW
import scipy.signal
import scipy.io

"""
read raw data and filter like in the Cole paper. Then analyze the waveform like in the cole paper. 
Does it make a difference to take the values of the raw data for peak analysis versus to take the filtered values?  
"""

path_to_file = '/Users/Jan/Dropbox/Master/LR_Kuehn/data/example_Cole2.mat'
save_folder = os.path.join(SAVE_PATH_DATA_BAROW, 'analysis')

d = scipy.io.loadmat(path_to_file)

# make new entries in the dict
result_dict = dict(sharpness={}, steepness={}, phase={})
frequ_range = 'beta'
result_dict['frequ_range'] = frequ_range

# define the list of conditions
n_conditions = 3
conditions = ['subj1to5_before_DBS', 'subj1to5_during_DBS', 'subj1to5_after_DBS']
# get the sampling rate
fs = d['fsample'][0][0]

n_subjects, n_samples = d['subj1to5_after_DBS'].shape

# prelocate result matrix: subjects * conditions
result_matrix_circ_mean = np.zeros((n_subjects, n_conditions))
result_matrix_esr = np.zeros((n_subjects, n_conditions))

# loop over conditions
for condition_idx, condition in enumerate(conditions):

    # loop over subjects
    for subject_idx, lfp_raw in enumerate(d[condition]):


        # take a look at the raw data
        # start = 1000
        # stop = start + 2000
        # plt.subplot(211)
        # plt.plot(lfp_raw[start:stop])
        #
        # # and at the psd
        # plt.subplot(212)
        # freqs, psd = ut.calculate_psd(y=lfp_raw, fs=fs, window_length=1024)
        # plt.plot(freqs[:40], psd[:40])
        # plt.show()

        # calculate circular mean vector amplitude
        # filter in beta range
        wn = np.array([13, 30]) / fs * 2
        # noinspection PyTupleAssignmentBalance
        b, a = scipy.signal.butter(2, wn, btype='bandpass')
        lfp_band = scipy.signal.filtfilt(b, a, lfp_raw)
        # lfp_band = ut.band_pass_filter(y=lfp_raw, fs=fs, band=[13, 30], plot_response=False)
        lfp_band = lfp_band[250:-250]
        lfp_band -= lfp_band.mean()

        # calculate the circular mean of the phases of the bandpass filtered signal
        analystic_signal = scipy.signal.hilbert(lfp_band)

        phase = np.unwrap(np.angle(analystic_signal))
        circular_mean_vector = np.mean(np.exp(1j * phase))
        circ_mean_angle = np.angle(circular_mean_vector)
        circ_mean_length = np.abs(circular_mean_vector)

        # save result
        result_matrix_circ_mean[subject_idx, condition_idx] = circ_mean_length

        # calculate extream sharpness ratio
        # identify time points of rising and falling zero-crossings:
        zeros_rising, zeros_falling, zeros = ut.find_rising_and_falling_zeros(lfp_raw)

        # find the peaks in between the zeros, USING THE RAW DATA!
        analysis_lfp = lfp_raw
        peaks, troughs, extrema = ut.find_peaks_and_troughs(analysis_lfp, zeros)

        # calculate peak sharpness:
        peak_sharpness = ut.calculate_peak_sharpness(analysis_lfp, peaks, fs=fs)
        trough_sharpness = ut.calculate_peak_sharpness(analysis_lfp, troughs, fs=fs)
        mean_peak_sharpness = np.mean(peak_sharpness)
        mean_trough_sharpness = np.mean(trough_sharpness)
        # extrema sharpness ratio, from the paper
        esr = np.max([mean_peak_sharpness / mean_trough_sharpness, mean_trough_sharpness / mean_peak_sharpness])

        # save the result
        result_matrix_esr[subject_idx, condition_idx] = esr

# plot the results
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.title('Extrema sharpness ratio (ESR) over conditions')
plt.plot(result_matrix_esr.mean(axis=0), '-o', label='esr-mean')
plt.legend()
plt.plot(result_matrix_esr.T, alpha=.5)
plt.ylabel('ESR and circular mean')
plt.xticks([], [])

plt.subplot(2, 1, 2)
plt.title('circurlar mean of phases over conditions')
plt.plot(result_matrix_circ_mean.mean(axis=0), '-o', label='circular mean, mean')
plt.plot(result_matrix_circ_mean.T, alpha=.5)
plt.ylabel('circular mean')
plt.legend()
plt.xticks(range(3), conditions)
plt.savefig(os.path.join(SAVE_PATH_FIGURES_BAROW, 'cole_example_subj1to5_IIR.pdf'))
# plt.show()
plt.close()