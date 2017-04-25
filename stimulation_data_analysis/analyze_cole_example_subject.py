import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from definitions import SAVE_PATH_FIGURES_BAROW, SAVE_PATH_DATA_BAROW
import scipy.io
import scipy.stats
import scipy.signal

"""
read raw data and filter like in the Cole paper. Then analyze the waveform like in the cole paper. 
Does it make a difference to take the values of the raw data for peak analysis versus to take the filtered values?  
"""

path_to_file = '/Users/Jan/Dropbox/Master/LR_Kuehn/bernadettes_results/example_Cole.mat'
save_folder = os.path.join(SAVE_PATH_DATA_BAROW, 'analysis')

d = scipy.io.loadmat(path_to_file)
# make new entries in the dict
result_dict = dict(sharpness={}, steepness={}, phase={})
frequ_range = 'beta'
result_dict['frequ_range'] = frequ_range


# define the list of conditions
conditions = ['subj11_before_DBS', 'subj11_during_DBS', 'subj11_after_DBS']
# get the sampling rate
fs = d['fsample'][0][0]

# look at the psd
#  f, psd = ut.calculate_psd(y=d['subj11_before_DBS'].squeeze(), fs=fsample, window_length=1024)
# plt.plot(f, psd)
# plt.show()

plt.figure(figsize=(10, 7))
raw_sample_length = 2  # in sec

for i, condition in enumerate(conditions):
    print('Condition {}'.format(condition))
    result_dict['sharpness'][condition] = {}
    result_dict['steepness'][condition] = {}
    result_dict['phase'][condition] = {}

    lfp = d[condition].squeeze() - np.mean(d[condition].squeeze())

    # remove line noise
    lfp = ut.band_stop_filter(y=lfp, fs=fs, band=[58, 62])

    # f, psd = ut.calculate_psd(y=data, fs=fsample, window_length=1024)
    # plt.plot(f, psd)
    # plt.show()

    # filter in beta range
    lfp_band = ut.band_pass_filter(y=lfp, fs=fs, band=[13, 30], plot_response=False)

    # identify time points of rising and falling zero-crossings:
    zeros_rising, zeros_falling, zeros = ut.find_rising_and_falling_zeros(lfp_band)

    # find the peaks in between the zeros, USING THE RAW DATA!
    analysis_lfp = lfp
    peaks, troughs, extrema = ut.find_peaks_and_troughs(analysis_lfp, zeros)

    # calculate peak sharpness:
    peak_sharpness = ut.calculate_peak_sharpness(analysis_lfp, peaks, fs=fs)
    trough_sharpness = ut.calculate_peak_sharpness(analysis_lfp, troughs, fs=fs)
    mean_peak_sharpness = np.mean(peak_sharpness)
    mean_trough_sharpness = np.mean(trough_sharpness)
    # extrema sharpness ratio, from the paper
    esr = np.max([mean_peak_sharpness / mean_trough_sharpness, mean_trough_sharpness / mean_peak_sharpness])

    # calculate the steepness
    rise_steepness, fall_steepness = ut.calculate_rise_and_fall_steepness(analysis_lfp, extrema)
    mean_rise_steepness = np.mean(rise_steepness)
    mean_fall_steepness = np.mean(fall_steepness)
    # rise decay steepness ratio
    rdsr = np.max([mean_rise_steepness / mean_fall_steepness, mean_fall_steepness / mean_rise_steepness])

    # calculate the circular mean of the phases of the bandpass filtered signal
    analystic_signal = scipy.signal.hilbert(lfp_band)
    phase = np.arctan2(np.imag(analystic_signal), np.real(analystic_signal)) + np.pi
    circular_mean_vector = np.mean(np.exp(1j * phase))
    circ_mean_angle = np.angle(circular_mean_vector)
    circ_mean_length = np.abs(circular_mean_vector)

    # filter in theta and beta to compare to pure sinusoidal shapes
    start = 5 * fs  # 5 sec offset
    stop = start + raw_sample_length * fs  # take 5 seconds only
    lfp_raw_sample = lfp[start:stop]

    result_dict['phase'][condition]['lfp_raw_sample'] = lfp_raw_sample

    # save to dict
    result_dict['sharpness'][condition]['trough_sharpness'] = trough_sharpness
    result_dict['sharpness'][condition]['peak_sharpness'] = peak_sharpness
    result_dict['sharpness'][condition]['trough_average'] = mean_trough_sharpness
    result_dict['sharpness'][condition]['peak_average'] = mean_peak_sharpness
    result_dict['sharpness'][condition]['esr'] = esr

    result_dict['steepness'][condition]['rise'] = rise_steepness
    result_dict['steepness'][condition]['fall'] = fall_steepness
    result_dict['steepness'][condition]['rise_average'] = mean_rise_steepness
    result_dict['steepness'][condition]['fall_average'] = mean_fall_steepness
    result_dict['steepness'][condition]['rdsr'] = rdsr

    result_dict['phase'][condition]['circular_mean_length'] = circ_mean_length
    result_dict['phase'][condition]['circular_mean_angle'] = circ_mean_angle



    # plot figure 1BC from the paper: histograms of the peak and trough values
    plt.subplot(1, 3, i + 1)
    plt.title(condition)
    plt.hist(np.log(result_dict['sharpness'][condition]['trough_sharpness']), label='trough', alpha=.5, bins=20)
    plt.hist(np.log(result_dict['sharpness'][condition]['peak_sharpness']), label='peak', alpha=.5, bins=20)
    plt.xlabel('log sharpness')

plt.legend()
plt.savefig(os.path.join(SAVE_PATH_FIGURES_BAROW, 'sharpness', 'cole_example_f1BC.pdf'))
# plt.show()

# plot figure 1D: esr over conditions
plt.figure(figsize=(8, 5))
plt.title('Extrema sharpness ratio (ESR) and circurlar mean of phases over conditions')
plt.plot([result_dict['sharpness'][c]['esr'] for c in conditions], '-o', label='esr')
plt.plot([result_dict['phase'][c]['circular_mean_length'] for c in conditions], '-o', label='circular mean')
plt.ylabel('ESR and circular mean')
plt.xticks(range(3), conditions)
plt.savefig(os.path.join(SAVE_PATH_FIGURES_BAROW, 'cole_example_f1D.pdf'))
# plt.show()

# plot the raw data and theta and beta waves
plt.figure(figsize=(10, 7))
plt.suptitle('{}sec lfp raw data with pure theta and beta waves'.format(raw_sample_length))
# determine theta and beta stuff
n_samples = raw_sample_length * fs
n_cycles_theta = raw_sample_length * 6
samples_theta = np.linspace(0, n_cycles_theta * 2 * np.pi, n_samples)
n_cycles_beta = raw_sample_length * 20
samples_beta = np.linspace(0, n_cycles_beta * 2 * np.pi, n_samples)
samples = np.arange(n_samples)

for i, c in enumerate(conditions):
    plt.subplot(3, 1, i + 1)
    # plot raw data
    lfp_raw_sample = result_dict['phase'][c]['lfp_raw_sample']
    plt.plot(samples, lfp_raw_sample)
    amplitude = 0.2 * abs(np.max(lfp_raw_sample) - np.min(lfp_raw_sample))
    # plot pure sinusoidal theta and beta
    plt.plot(samples, amplitude * np.sin(samples_theta), label='theta', alpha=.7)
    plt.plot(samples, amplitude * np.sin(samples_beta), label='beta', alpha=.7)
    plt.title(c)
    plt.ylabel('raw lfp [ $\mu V$ ]')
    if i < len(conditions) - 1:
        plt.xticks([], [])

plt.xlabel('time [ms]')
plt.savefig(os.path.join(SAVE_PATH_FIGURES_BAROW, 'cole_example_rawLFP_vs_sinusoidal.pdf'))
plt.show()

# save data
# ut.save_data(data_dict=result_dict,
#              filename='cole_example_subject_sharpness_steepness_{}.p'.format(frequ_range),
#              folder=save_folder)

