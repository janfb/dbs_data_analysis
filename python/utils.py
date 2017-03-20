from scipy.io import loadmat
from definitions import DATA_PATH, SAVE_PATH_DATA
import os
import scipy.signal
import numpy as np
import sys
import pickle
from scipy.optimize import curve_fit


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
    full_path_to_file = os.path.join(folder, filename)
    with open(full_path_to_file, 'wb') as outfile:
        pickle.dump(data_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def load_data_analysis(filename, data_folder=SAVE_PATH_DATA):
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


def band_pass_filter(y, fs, band=np.array([4, 45]), order=5, btype='bandpass'):
    """
    Band-pass filter in a given frequency band
    :param y: the time series
    :param fs: the sampling rate
    :param order: the filter order, degault 5
    :param band: the frequency band to remain
    :return: the filtered time series
    """
    wn = band / fs * 2
    # noinspection PyTupleAssignmentBalance
    b, a = scipy.signal.butter(order, wn, btype=btype)
    return scipy.signal.filtfilt(b, a, y)


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


def calculate_spectrogram(y, fs):
    return scipy.signal.spectrogram(y, fs=fs, nperseg=1024)


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
    return indices[peak_idx], frequs_band, psd_band


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
