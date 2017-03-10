from scipy.io import loadmat
from definitions import DATA_PATH
import os
import scipy.signal
import numpy as np
import sys


def load_data(filename='spmeeg_1.mat', data_folder=DATA_PATH):
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
    wn = np.array([48, 52]) / fs * 2
    b, a = scipy.signal.butter(order, wn, btype='bandstop')
    return scipy.signal.lfilter(b, a, y)


def calculate_psd(y, fs):
    """
    Calculate the power spectral density as in the dystonia rest paper: 1024 samples per epoch, no overlap,
    linear detrending.
    :param y: time series
    :param fs: sampling rate
    :return f: vector of frequencies
    :return psd: vector of psd
    """
    f, psd = scipy.signal.welch(y, fs=fs, window='hamming', nperseg=1024, noverlap=0, detrend='linear')
    return f, psd


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


def find_peak_in_band(frequs, psd, band):
    """
    Find the maximum peak in a given range of a power spectrum
    :param frequs: the frequency vector
    :param psd: the psd vector, same length
    :param band: a list or sequence with the range, e.g., [4, 12]
    :return: the index of the peak, the masked frequs vector, the masked psd vector
    """
    mask = get_array_mask(frequs > band[0], frequs < band[1])
    [maxtab, mintab] = peakdet(psd[mask], delta=1e-6)
    # get the indices of all maxima
    indices = np.array(maxtab[:, 0], int)
    # remove the zero index if in there
    indices = indices[indices > 0]
    # select the maximum peak
    peak_idx = np.argmax(psd[indices])
    return indices[peak_idx], frequs[mask], psd[mask]


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
