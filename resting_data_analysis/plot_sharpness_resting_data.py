import os
import numpy as np
import utils as ut
from definitions import SAVE_PATH_DATA, SAVE_PATH_FIGURES
import matplotlib.pyplot as plt

"""
for every subject make a plot of histograms of sharpness of peaks and troughs
"""
data_folder = os.path.join(SAVE_PATH_DATA, 'sharpness_steepness')

band = 'theta'
# read all files in the data folder
file_list = [f for f in os.listdir(data_folder) if f.endswith(band + '.p')]

# collect all values of esr and rdsr over subjects and channels in a single list
esr_list = []
rdsr_list = []

# for every subject file
for sub, sub_file in enumerate(file_list):
    # load data
    d = ut.load_data_analysis(sub_file, data_folder=data_folder)
    frequ_range = d['frequ_range']
    print('Loading subject file {}'.format(sub_file))

    plt.figure(figsize=(10, 7))
    plt.subplot(2, 3, 1)
    plt.ylabel('cycle count')

    channels = d['channels']
    # for all three conditions
    for i, channel in enumerate(channels):
        c = channel[0]
        print('Condition {}'.format(c))
        n_bins = 30

        plt.subplot(2, 3, i + 1)
        plt.title(c)
        troughs = d['sharpness'][c]['trough_sharpness']
        mi, ma = np.min(troughs), np.max(troughs)
        bins = np.power(10, np.linspace(np.log10(mi), np.log10(ma), n_bins))
        plt.hist(troughs, label='troughs', alpha=.5,
                 bins=n_bins)
        # plt.gca().set_xscale("log")

        peaks = d['sharpness'][c]['peak_sharpness']
        mi, ma = np.min(peaks), np.max(peaks)
        bins = np.power(10, np.linspace(np.log10(mi), np.log10(ma), n_bins))
        plt.hist(peaks, label='peaks', alpha=.5,
                 bins=n_bins)
        if i > 2:
            plt.xlabel('peak sharpness')

        # collect ratios in list
        esr_list.append(d['sharpness'][c]['esr'])
        rdsr_list.append(d['steepness'][c]['rdsr'])

    plt.legend()
    plt.suptitle('Sharpness of the peaks and troughs in the {} waveform'.format(frequ_range))
    filename_figure = '{}_subject_{}_sharpness_histogram.pdf'.format(frequ_range, d['subject_number'])
    # plt.savefig(os.path.join(SAVE_PATH_FIGURES, 'sharpness_steepness', filename_figure))
    # plt.show()
    plt.close()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(x=esr_list, bins=10)
plt.title('Estrema sharpness ratio')
plt.xlabel('sharpness ratio')

plt.subplot(1, 2, 2)
plt.hist(x=rdsr_list, bins=10)
plt.title('Rise decay steepness ratio')
plt.xlabel('steepness ratio')

plt.ylabel('count')
plt.suptitle('Histograms over all subjects and channels')
figure_name = 'esr_rdsr_histograms.pdf'
plt.savefig(os.path.join(SAVE_PATH_FIGURES, figure_name))
plt.show()
plt.close()


