from scipy.io import loadmat
from definitions import DATA_PATH, SAVE_PATH_DATA
import os
import scipy.signal
import numpy as np
import sys
import pickle
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def low_pass_filter(y, fs, cutoff=200, numtaps=250):
    """
    Low pass filter using the window method for FIR filters 
    :param y: time series 
    :param fs: sampling rate
    :param numtaps: filter length      
    :param cutoff: cutoff frequency in Hz
    :return: filtered time series  
    """
    nyq = fs / 2
    cut_off_normalized = cutoff / nyq
    coefs = scipy.signal.firwin(numtaps=numtaps, cutoff=cut_off_normalized)
    return scipy.signal.filtfilt(coefs, 1., y)

def select_time_periods_of_high_power(data, fs=1000, band=np.array([13, 30])):

    # calculate spectrogram
    nyq = fs / 2
    f, t, Sxx = scipy.signal.spectrogram(x=data, fs=fs, window='hamming', nperseg=512, noverlap=256)
    # get band mask
    mask = get_array_mask(f > band[0], f < band[1])
    # extract mean power in the range
    power = Sxx[mask, :].mean(axis=0)
    # smooth the power to select broader bursts
    power_smooth = scipy.signal.filtfilt(scipy.signal.firwin(numtaps=int(data.size / 1000), cutoff=50 / nyq), 1., power)
    # define a threshold to select the time periods
    threshold = power_smooth.mean() + 1 * power_smooth.std()

    # resample the arrays to meet data size
    t2 = np.arange(0, data.size / fs, 1 / fs)
    power_smooth2 = np.zeros_like(t2)
    mask = get_array_mask(t2 <= t[-1])

    power_smooth2[mask] = np.interp(t2[mask], t, power_smooth)
    power_smooth2[~mask] = power_smooth[-1]

    # return a mask of time periods, ignore first 2 seconds of signal
    burst_mask = get_array_mask(power_smooth2 >= threshold)
    return t2, burst_mask


def calculate_circular_mean(phases):
    """
    Calculate the circular mean vector angle and vector length from the phases. the length of the mean phase vector is 0
    if the phases are distributed uniformly and 1 if they are all the same. The angle of the mean phase vector is the
    average direction the phases are pointing towards. Its meaning depends on the length of the mean phase vector, e.g., 
    it is most informative if the mean phase vector is large. 
    :param phases: vector of phases 
    :return: angle of the mean phase vector, length of the mean phase vector. 
    """
    circular_mean_vector = np.mean(np.exp(1j * phases))
    circ_mean_angle = np.angle(circular_mean_vector)
    circ_mean_length = np.abs(circular_mean_vector)
    return circ_mean_angle, circ_mean_length


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
    Use cwt to find the indices of the minima and maxima of a array given on the basis of an array of zero crossings.
    :param y: time series
    :param zeros: array of indices of zero crossing
    :return: peaks, trough, arrays
    """
    # find the peaks in between the zeros
    peak_indices = []
    trough_indices = []
    extrema_indices = []

    # zero mean the data to find the zero crossings
    y -= y.mean()

    # for every zero
    idx = 0
    while idx < zeros.shape[0] - 1:
        # get the sub array
        sub_array = y[zeros[idx]:zeros[idx + 1]]
        # if sub array to short, skip this zero
        skipped = False
        if sub_array.size < 10:
            sub_array = y[zeros[idx]:zeros[idx + 2]]
            skipped = True
        # center it
        sub_array -= np.mean(sub_array)

        # find peak and trough
        widths = np.arange(1, sub_array.size / 2)  # kernel size
        peakind = scipy.signal.find_peaks_cwt(vector=sub_array, widths=widths)
        troughind = scipy.signal.find_peaks_cwt(vector=-sub_array, widths=widths)
        # hack if no extrema was found
        if len(peakind) == 0:
            peakind = [np.argmax(sub_array)]
        if len(troughind) == 0:
            troughind = [np.argmin(sub_array)]

        peak_idx = peakind[np.argmax(sub_array[peakind])]
        trough_idx = troughind[np.argmin(sub_array[troughind])]

        # select the index with the larger signal amplitude
        if abs(sub_array[peak_idx]) > abs(sub_array[trough_idx]):
            # get the global idx
            extrema_idx = peak_idx + zeros[idx]
            # add it to the corresp. list
            peak_indices.append(extrema_idx)
        else:
            extrema_idx = trough_idx + zeros[idx]
            trough_indices.append(extrema_idx)

        # add to overall list
        # extrema_idx = np.argmax(abs(sub_array)) + zeros[idx]
        extrema_indices.append(extrema_idx)
        idx += 1
        if skipped:
            idx += 1

    return np.array(peak_indices), np.array(trough_indices), np.array(extrema_indices)


def advanced_peak_search(sub_array, verbose=True):
    """
    Do an advanced search for peaks by looking at the derivate of the signal 
    :param sub_array: time series 
    :return: 
    """
    sub_array -= sub_array.mean()
    # find peak and trough
    out = peakdet(v=sub_array, delta=1e-6)
    # join the extrema
    try:
        extrema = np.vstack((out[0], out[1]))
        if out[0].size == 0:
            extrema = out[1]
        elif out[1].size == 0:
            extrema = out[0]
        elif out[0].size == 0 and out[1].size == 0:
            if not verbose:
                print('No extrema were found, using first idx')
            return 0
    # exclude zero indices
        if extrema.size > 2:
            extrema = extrema[extrema[:, 0] > 0, :]
        # find extremum with largest amplitude
        extremum_idx = extrema[np.argmax(abs(extrema[:, 1])), 0]
    except:
        if not verbose:
            print('No extrema were found, using first idx')
        return 0
    return extremum_idx


def find_peaks_and_troughs_cole(y, zeros, rising_zeros, falling_zeros):
    """
    Find peaks and troughs as defined in the cole paper.
      
        "the time point of maximal voltage between a rising zero-crossing and a subsequent falling zero-crossing was
        defined as the peak"
        
    When needed, improve the search by looking at the derivative.
    :param y: time series 
    :param rising_zeros: indices of rising zero crossing 
    :param falling_zeros: ...
    :return: indices of peaks, troughs, extrema 
    """
    peak_indices = []
    trough_indices = []
    extrema_indices = []
    y -= y.mean()
    # it should first look for maximum if rising zero comes first
    peak_trough_factor = 1. if rising_zeros[0] < falling_zeros[0] else -1.

    for idx, zero in enumerate(zeros[:-1]):
        sub_array = peak_trough_factor * y[zeros[idx]:zeros[idx + 1]]
        extrema_idx = np.argmax(sub_array)

        # if the first index or the last was selected we should look more closely using the derivative
        if (extrema_idx == 0 or extrema_idx == sub_array.size - 1) and sub_array.size > 10:
            # do an advanced search
            extrema_idx = int(advanced_peak_search(sub_array))

        extrema_idx += zeros[idx]
        if y[extrema_idx] > 0:
            peak_indices.append(extrema_idx)
        else:
            trough_indices.append(extrema_idx)
        # add to overall list
        extrema_indices.append(extrema_idx)
        # alternate extrema
        peak_trough_factor *= -1

    return np.array(peak_indices, dtype=int), np.array(trough_indices, dtype=int), np.array(extrema_indices, dtype=int)


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
    zeros = np.where(y_sign_change != 0)[0]

    # double check the zeros and correct if possible
    new_zeros = []
    for i in range(1, zeros.shape[0] - 1):
        # for every detected zero index check if neighbors are better and add them if yes. keep it otherwise
        if abs(y[zeros[i] - 1]) < abs(y[zeros[i]]):
            new_zeros.append(zeros[i] - 1)
        elif abs(y[zeros[i] + 1]) < abs(y[zeros[i]]):
            new_zeros.append(zeros[i] + 1)
        else:
            new_zeros.append(zeros[i])
    zeros = np.unique(np.array(new_zeros))

    if y[zeros[0] - 1] > 0:  # falling zero comes first
        zeros_falling = zeros[0::2]
        zeros_rising = zeros[1::2]
    else:  # rising zero comes first
        zeros_falling = zeros[1::2]
        zeros_rising = zeros[0::2]

    # # debug plot
    # # zeros = new_zeros
    # upto = 50
    # plt.plot(zeros[:upto], y[zeros][:upto], '*', markersize=10)
    # plt.plot(y[:zeros[upto]], 'o')
    # plt.axhline(y=0)
    # plt.title('Error = {}'.format(np.sum(np.abs(y[zeros]))))
    # plt.show()
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


def calculate_cole_ratios(lfp_pre, lfp_band, fs):
    # COLE ANALYSIS
    # identify time points of rising and falling zero-crossings:
    zeros_rising, zeros_falling, zeros = find_rising_and_falling_zeros(lfp_band)

    # find the peaks in between the zeros, USING THE RAW DATA!
    analysis_lfp = lfp_pre
    peaks, troughs, extrema = find_peaks_and_troughs_cole(analysis_lfp,
                                                             zeros=zeros,
                                                             rising_zeros=zeros_rising,
                                                             falling_zeros=zeros_falling)

    peak_sharpness = calculate_peak_sharpness(analysis_lfp, peaks, fs=fs)
    trough_sharpness = calculate_peak_sharpness(analysis_lfp, troughs, fs=fs)
    mean_peak_sharpness = np.mean(peak_sharpness)
    mean_trough_sharpness = np.mean(trough_sharpness)
    # extrema sharpness ratio, from the paper
    esr = np.max([mean_peak_sharpness / mean_trough_sharpness, mean_trough_sharpness / mean_peak_sharpness])

    # calculate the steepness
    rise_steepness, fall_steepness = calculate_rise_and_fall_steepness(analysis_lfp, extrema)
    mean_rise_steepness = np.mean(rise_steepness)
    mean_fall_steepness = np.mean(fall_steepness)
    # rise decay steepness ratio
    rdsr = np.max([mean_rise_steepness / mean_fall_steepness, mean_fall_steepness / mean_rise_steepness])

    return esr, rdsr


def exclude_outliers(x, y, n=2):
    """
    Exclude outliers in x and y. The criterion is just the distance in n stds from the sample mean. If a value is extreme 
     in x and y it is excluded. 
    :param x: 
    :param y: 
    :param n: std factor 
    :return: x y with identified outliers excluded
    """
    mask_x = x > (x.mean() + n * x.std())  # extreme x values
    mask_y = y > (y.mean() + n * y.std())  # extreme y values

    mask = np.logical_not(np.logical_or(mask_x, mask_y))  # NOT outliers
    x_out = x[np.logical_not(mask)]
    y_out = y[np.logical_not(mask)]

    return x[mask], y[mask], x_out, y_out