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
file_list = [f for f in os.listdir(data_folder) if f.endswith('beta.p') and f.startswith('subject')]

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
        condition = condition_order[i]
        print('Condition {}'.format(condition))
        n_bins = 30

        plt.subplot(1, 3, i + 1)
        troughs = d['sharpness'][condition]['trough_sharpness']
        peaks = d['sharpness'][condition]['peak_sharpness']
        plt.title('{}'.format(condition))
        hist_troughs, bins = np.histogram(np.log(troughs), bins=30)
        plt.hist(np.log(troughs), label='trough', alpha=.5, bins=bins)
        plt.hist(np.log(peaks), label='peak', alpha=.5, bins=bins)
        plt.xlabel('log sharpness')

    plt.legend()
    plt.suptitle('Sharpness of the peaks and troughs in the {} waveform'.format(frequ_range))
    filename_figure = '{}_subject_{}_sharpness_histogram.pdf'.format(frequ_range, d['number'])
    plt.savefig(os.path.join(SAVE_PATH_FIGURES_BAROW, 'sharpness', filename_figure))
    # plt.show()
    plt.close()
