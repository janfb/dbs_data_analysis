import os
import numpy as np
import utils as ut
from definitions import SAVE_PATH_DATA_BAROW, SAVE_PATH_FIGURES_BAROW
import matplotlib.pyplot as plt

"""
for every subject make a plot of histograms of sharpness of peaks and troughs
"""
data_folder = os.path.join(SAVE_PATH_DATA_BAROW, 'analysis')

# read all files in the data folder
file_list = [f for f in os.listdir(data_folder) if f.endswith('theta.p')]

# for every subject file
for sub, sub_file in enumerate(file_list):
    # load data
    d = ut.load_data_analysis(sub_file, data_folder=data_folder)
    frequ_range = d['frequ_range']
    print('analysing subject file {}'.format(sub_file))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.ylabel('cycle count')

    condition_order = ['rest', 'stim', 'poststim']
    conditions = d['conditions']
    # for all three conditions
    for i in range(len(conditions)):
        c = condition_order[i]
        print('Condition {}'.format(c))
        n_bins = 30

        plt.subplot(1, 3, i + 1)
        plt.title(c)
        troughs = d['sharpness'][c]['trough_sharpness']
        mi, ma = np.min(troughs), np.max(troughs)
        bins = np.power(10, np.linspace(np.log10(mi), np.log10(ma), n_bins))
        plt.hist(troughs, label='troughs', alpha=.5,
                 bins=bins)
        plt.gca().set_xscale("log")

        peaks = d['sharpness'][c]['peak_sharpness']
        mi, ma = np.min(peaks), np.max(peaks)
        bins = np.power(10, np.linspace(np.log10(mi), np.log10(ma), n_bins))
        plt.hist(peaks, label='peaks', alpha=.5,
                 bins=bins)
        plt.gca().set_xscale("log")
        # plt.hist(d['sharpness'][c]['peak_sharpness'], label='peaks', alpha=.5)

    plt.legend()
    plt.suptitle('Sharpness of the peaks and troughs in the {} waveform'.format(frequ_range))
    # plt.show()
    filename_figure = 'subject_{}_sharpness_histogram_{}.pdf'.format(d['number'], frequ_range)
    plt.savefig(os.path.join(SAVE_PATH_FIGURES_BAROW, 'sharpness', filename_figure))






