from scipy.io import loadmat
from definitions import DATA_PATH, SAVE_PATH_DATA
import os
import scipy.signal
import numpy as np
import sys
import pickle
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def calculate_rise_and_fall_steepness(y, extrema):
    """
    Calculate the rise steepness between trough and peaks and the fall steepness between peaks and troughs from the time
    series and a list of indices of extrema.
    :param y: time series
    :param extrema: index list of extrema
    :return: array of rise steepness, array of fall steepness
    """
    # calculate absolute instantaneous first derivative:
    lfp_diff = np.abs(np.diff(y))
    rise_steepness = []
    fall_steepness = []

    for idx in range(extrema[:-1].size):
        # check whether it is trough or a peak
        if y[extrema[idx]] < 0:  # case trough
            # calculate rise steepness: max slope between trough and peak
            max_slope = np.max(lfp_diff[extrema[idx]:extrema[idx + 1]])  # the next idx is a peak
            rise_steepness.append(max_slope)
        else:
            # case peak
            max_slope = np.max(lfp_diff[extrema[idx]:extrema[idx + 1]])  # the next idx is a trough
            fall_steepness.append(max_slope)

    return np.array(rise_steepness), np.array(fall_steepness)


def find_peaks_and_troughs(y, zeros):
    """
    Find the indices of the minima and maxima of a array given on the basis of an array of zero crossings
    :param y: time series
    :param zeros: array of indices of zero crossing
    :return: peaks, trough, arrays
    """
    # find the peaks in between the zeros
    peaks = []
    troughs = []
    extrema = []

    # for every zero
    for idx in range(zeros[:-1].size):
        # check whether we will have a min or max
        sub_array = y[zeros[idx]:zeros[idx + 1]]
        mean = np.mean(sub_array)
        # if max, add to peak array, make sure to add the idx offset
        if mean > 0:
            peak_idx = np.argmax(sub_array) + zeros[idx]
            peaks.append(peak_idx)
            extrema.append(peak_idx)
        # else add to trough array
        else:
            trough_idx = np.argmin(sub_array) + zeros[idx]
            troughs.append(trough_idx)
            extrema.append(trough_idx)

    return np.array(peaks), np.array(troughs), np.array(extrema)


def calculate_peak_sharpness(y, peaks, fs):
    """
    Calculate the sharpness of peaks given the time series, peak indices and the sampling rate
    :param y: time series, array
    :param peaks: indices of peaks, array
    :param fs: sampling rate
    :return: array of peak sharpness, same size as peaks
    """
    sharpness = np.zeros_like(peaks, dtype=np.float)
    # samples for ms
    samples_per_ms = fs / 1000

    # for every peak
    for idx, peak_idx in enumerate(peaks):
        # get indices +-5 ms around the peak
        precede_idx = int(peak_idx - 5 * samples_per_ms)
        follow_idx = int(peak_idx + 5 * samples_per_ms)

        # avoid index error
        if y.size <= follow_idx:
            follow_idx = -1
        if precede_idx < 0:
            precede_idx = 0

        # apply formula from the cole paper
        sharpness[idx] = 0.5 * (np.abs(y[peak_idx] - y[precede_idx]) + np.abs(y[peak_idx] - y[follow_idx]))

    return sharpness


def find_rising_and_falling_zeros(y):
    """
    Find the zero crossings, separated in rising zeros and falling zeros
    :param y: time series
    :return: rising zeros indices, falling zeros indices
    """
    y_sign = np.sign(y)
    y_sign_change = np.diff(y_sign)
    zeros_falling = np.where(y_sign_change == -2)[0]
    zeros_rising = np.where(y_sign_change == 2)[0]
    zeros = np.where(y_sign_change != 0)[0]
    return zeros_rising, zeros_falling, zeros


def coherency(x, y, fs, window_length=1024, **kwargs):
    """
    Calculate the cohenrencY between values in x and in y. It is defined as the normalized cross spectral density:
    Cij(f) = Sij(f) / (Sii(f)*Sjj(f))^1/2
    The coherence is then defined as the absolute value of that: Coh_ij(f) = abs(Cij(f))
    :param x: one time series
    :param y: another time series
    :param fs: common sampling rate
    :param window_length: window length for Welch method
    :param kwargs: addition arguments for Welch method
    :return: frequency vector f and complex coherence vector
    """
    f, pxx = scipy.signal.welch(x, fs=fs, window='hamming', nperseg=window_length, **kwargs)
    _, pxy = scipy.signal.csd(x, y, fs=fs, window='hamming', nperseg=window_length, **kwargs)
    _, pyy = scipy.signal.welch(y, fs=fs, window='hamming', nperseg=window_length, **kwargs)

    return f, pxy / np.sqrt(pxx * pyy)


def standard_error(y, axis=0):
    """
    Calculate the standard error of y: se = std(y) / sqrt(n)
    :param y: array - likes, at most 2 dim
    :param axis: axis on which to calculate the standard error
    :return: se of y
    """
    assert (axis == 0 or axis == 1), ' axis can only be 0 or 1'
    if y.ndim == 1:
        return np.std(y) / np.sqrt(y.shape[0])
    else:
        return np.std(y, axis=axis) / np.sqrt(y.shape[axis])


def remove_1f_component(f, psd):
    """
    Remove the 1/f component of a given power spectral density. This is done by linearization of the spectrum via log-
     transformation, then fitting a line from the 1-4Hz range to the 35-40Hz range and subtracting the line from the
     linear spectrum and finally, by transformation back to the standard representation.
    :param f: vector of frequency samples of the psd
    :param psd: vector of psd values corresponding to the frequencies in f
    :return: f and psd where the 1/f component has been subtracted
    """
    # do log log transform, remove 1/f component via least squares fit
    psd_log = np.log10(psd)
    f_log = np.log10(f)
    # set the first entry to large negative number  because it becomes infinitiy in log
    if not np.isfinite(f_log[0]):
        f_log[0] = -10

    # take only the data from 1-4 and 35-40 for fitting
    mask = np.logical_or(get_array_mask(f > 1, f < 4), get_array_mask(f > 35, f < 40))
    # fit a line
    line = lambda a, b, x: a * x + b
    a, b = curve_fit(line, xdata=f_log[mask], ydata=psd_log[mask])[0]
    one_over_f_component = line(f_log, a, b)
    # subtract that from the log transformed psd
    psd_log -= one_over_f_component
    # transform back
    psd = np.power(10., psd_log)
    f = np.power(10., f_log)
    return f, psd


def save_data(data_dict, filename, folder=SAVE_PATH_DATA):
    """
    Save file with pickle
    :param data_dict: dictionary holding the data
    :param filename:
    :param folder:
    :return:
    """
    full_path_to_file = os.path.join(folder, filename)
    with open(full_path_to_file, 'wb') as outfile:
        pickle.dump(data_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_data_analysis(filename, data_folder=SAVE_PATH_DATA):
    """
    Load file with pickle
    :param filename:
    :param data_folder:
    :return: data dictionary
    """
    full_path = os.path.join(data_folder, filename)
    return pickle.load(open(full_path, 'rb'))


def load_data_spm(filename='spmeeg_1.mat', data_folder=DATA_PATH):
    """
    Load a file of the original spm data files that have been converted to readable python format
    :param filename: the filename, 1 by default
    :param data_folder: the folder holding all the data
    :return: the dictionary containing the lfp and meta data
    """
    return loadmat(os.path.join(data_folder, filename))


def remove_50_noise(y, fs, order=2):
    """
    Remove the 50Hz humming noise from a time series y
    :param y: the time series
    :param fs: the sampling rate
    :param order: the filter order, degault 2
    :return: the time series, 50Hz noise filtered out
    """
    wn = np.array([49, 51]) / fs * 2
    # noinspection PyTupleAssignmentBalance
    b, a = scipy.signal.butter(order, wn, btype='bandstop')
    y1 = scipy.signal.lfilter(b, a, y)
    # filter out super harmonics
    wn = np.array([99, 101]) / fs * 2
    # noinspection PyTupleAssignmentBalance
    b, a = scipy.signal.butter(order, wn, btype='bandstop')
    y2 = scipy.signal.lfilter(b, a, y1)
    wn = np.array([149, 151]) / fs * 2
    # noinspection PyTupleAssignmentBalance
    b, a = scipy.signal.butter(order, wn, btype='bandstop')
    return scipy.signal.lfilter(b, a, y2)


def band_pass_filter(y, fs, band=np.array([4, 45]), pass_zero=False, plot_response=False):
    """
    Band-pass filter in a given frequency band
    :param y: the time series
    :param fs: the sampling rate
    :param band: the frequency band to remain
    :param plot_response: flag for plotting the filter response to double check the filter
    :param pass_zero: whether to include DC: False makes it a bandpass filter, True a bandstop filter
    :return: the filtered time series, the frequencies and the frequency response of the filter
    """
    # get Nyquist frequency
    nyq = fs / 2.
    # get the number of samples per ms
    samples_per_ms = fs / 1000.
    # get the cycle length in samples
    cycle_length = int(1000 / band[0] * samples_per_ms)
    # heuristics: filter order should triple the maximal cycle length in samples
    numtaps = 3 * cycle_length
    # make numtaps even
    if not numtaps % 2:
        numtaps += 1
    # design a FIR bandpass filter using the window method. pass-zero makes it a bandpass filter by enforcing the DC
    # response to be 0
    coefs = scipy.signal.firwin(numtaps=numtaps, cutoff=(band[0], band[1]), window='hamming', nyq=nyq,
                                pass_zero=pass_zero)

    # if needed, plot the filter response
    if plot_response:
        plot_filter_response(coefs, nyq, band)
    # return the filtered signal
    return scipy.signal.filtfilt(coefs, 1., y)


def plot_filter_response(coefs, nyq, band):
    """
    Plot the filter response in the frequency spectrum
    :param coefs: filter coefficients
    :param nyq: Nyquist frequency
    :param band: frequency band of interest
    """
    # calculate the response
    freq, response = scipy.signal.freqz(coefs)
    # plotting
    upto = int(band[1] + 30)
    f2 = plt.figure()
    plt.semilogy((nyq * freq / np.pi)[:upto], np.abs(response)[:upto], label='firs')
    plt.xlim([0, upto])
    plt.title('Frequency response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.show()


def band_stop_filter(y, fs, band=np.array([48, 52]), plot_response=False):
    """
    Design and apply a bandstop filter to remove a specific frequency content, e.g., the 50Hz noise component
    :param y: time series
    :param fs: sampling rate
    :param band: band to be removed
    :return: filtered signal
    """
    # just call the bandpass filter method with pass_zero=True. then the DC will be included and the filter response
    # will be such that excludes the range in 'band'
    return band_pass_filter(y, fs, band, pass_zero=True, plot_response=plot_response)


def calculate_psd(y, fs, window_length=1024):
    """
    Calculate the power spectral density as in the dystonia rest paper: 1024 samples per epoch, no overlap,
    linear detrending.
    :param window_length: nperseg for welch
    :param y: time series
    :param fs: sampling rate
    :return f: vector of frequencies
    :return psd: vector of psd
    """
    f, psd = scipy.signal.welch(y, fs=fs, window='hamming', nperseg=window_length, noverlap=window_length / 2,
                                detrend='linear')
    return f, psd


def calculate_psd_epoching(y, fs, epoch_length=1024):
    """
    Calculate the PSD using scipy and welch method after epoching it into epochs of epoch_length samples
    :param y: the signal vector
    :param fs: sampling rate
    :param epoch_length:
    :return: f vector, psd vector
    """
    # calculate the largest number of epochs to be extracted
    n_epochs = int(np.floor(y.shape[0]) / epoch_length)
    idx = n_epochs * epoch_length
    # cut off the vector sample that do not fit
    y = y[:idx]
    # reshape into epochs in rows
    y = np.reshape(y, (n_epochs, epoch_length))
    # calculate the psd of all epochs
    f, psds = scipy.signal.welch(y, fs=fs, window='hamming', nperseg=epoch_length)
    return f, psds.mean(axis=0)


def calculate_spectrogram(y, fs, window_length=1024):
    """
    Calculate the time-frequency decomposition of y with sampling rate fs using scipy.signal.spectrogram
    :param y:
    :param fs:
    :return:
    """
    return scipy.signal.spectrogram(y, fs=fs, nperseg=window_length)


def get_array_mask(cond1, *args):
    """
    build a mask of several and conditions
    :param cond1: first condition
    :param args: optional additional conditions
    :return: conjunction of all conditions in a logical array
    """
    mask = cond1
    for arg in args:
        mask = np.logical_and(mask, arg)
    return mask


def find_peak_in_band(frequs, psd, band, linearize=False):
    """
    Find the maximum peak in a given range of a power spectrum
    :param frequs: the frequency vector
    :param psd: the psd vector, same length
    :param band: a list or sequence with the range, e.g., [4, 12]
    :return: the index of the peak, the masked frequs vector, the masked psd vector
    """

    mask = get_array_mask(frequs > band[0], frequs < band[1])
    psd_band = psd[mask]
    frequs_band = frequs[mask]

    # remove 1/f component if specified
    if linearize:
        frequs_linear, psd_linear = remove_1f_component(frequs, psd)
        frequs_search, psd_search = frequs_linear[mask], psd_linear[mask]
    else:
        frequs_search, psd_search = frequs_band, psd_band

    [maxtab, _] = peakdet(v=psd_search, delta=1e-10)

    # get the indices of all maxima
    try:
        indices = np.array(maxtab[:, 0], int)
    except IndexError:
        indices = np.array([0])
        print('Cound not find a peak, taking the first index instead!')

    # remove the zero index if in there
    if indices[0] == 0 and indices.shape[0] > 1:
        indices = indices[1:]

    # select the maximum peak
    peak_idx = np.argmax(psd_search[indices])

    # get the amplitude of that peak
    surround_mask = get_array_mask(frequs_search > (frequs_search[indices[peak_idx]] - 1.5),
                                   frequs_search < (frequs_search[indices[peak_idx]] + 1.5))
    peak_amp = np.mean(psd_search[surround_mask])

    # if the zero index was selected, set amplitude to zero to avoid this channel for later analysis
    if indices[0] == 0:
        peak_amp = 0
    return indices[peak_idx], peak_amp, frequs_band, psd_band


def peakdet(v, delta, x=None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)
