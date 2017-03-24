import os
import numpy as np
import utils as ut
from definitions import SAVE_PATH_DATA_BAROW, SAVE_PATH_FIGURES_BAROW
import matplotlib.pyplot as plt

"""
also add a plot of log esr of stim vs. rest, rest vs. poststim, stim vs. poststim
same for rdsr
"""
data_folder = os.path.join(SAVE_PATH_DATA_BAROW, 'analysis')

# read all files in the data folder
file_list = [f for f in os.listdir(data_folder) if f.endswith('theta.p')]

esr = dict(rest=np.zeros(len(file_list)), stim=np.zeros(len(file_list)), poststim=np.zeros(len(file_list)))
rdsr = dict(rest=np.zeros(len(file_list)), stim=np.zeros(len(file_list)), poststim=np.zeros(len(file_list)))

frequ_range = None

# for every subject file
for sub, sub_file in enumerate(file_list):
    # load data
    d = ut.load_data_analysis(sub_file, data_folder=data_folder)
    frequ_range = d['frequ_range']
    # print('analysing subject file {}'.format(sub_file))

    condition_order = ['rest', 'stim', 'poststim']
    conditions = d['conditions']
    # for all three conditions
    for i in range(len(conditions)):
        c = condition_order[i]
        # print('Condition {}'.format(c))

        esr[c][sub] = d['sharpness'][c]['esr']
        rdsr[c][sub] = d['steepness'][c]['rdsr']

plt.figure(figsize=(10, 7))
plot_list = [['rest', 'stim'], ['stim', 'poststim'], ['rest', 'poststim']]
for i, plot_tuple in enumerate(plot_list):
    plt.subplot(1, 3, i + 1)
    plt.plot(esr[plot_tuple[0]], esr[plot_tuple[1]], 'o')
    maximum = np.max([esr[plot_tuple[0]], esr[plot_tuple[1]]]) + 0.02
    plt.axis([1, maximum, 1, maximum])
    plt.xlabel('esr, {}'.format(plot_tuple[0]))
    plt.ylabel('esr, {}'.format(plot_tuple[1]))
    line = np.linspace(1, maximum, 10)
    plt.plot(line, line)
    plt.title('{} vs. {}'.format(plot_tuple[0], plot_tuple[1]))

plt.suptitle('Extrema sharpness ratio')
# plt.show()
filename_figure = 'esr_comparison.pdf'
plt.savefig(os.path.join(SAVE_PATH_FIGURES_BAROW, 'sharpness', filename_figure))

plt.figure(figsize=(10, 7))
plot_list = [['rest', 'stim'], ['stim', 'poststim'], ['rest', 'poststim']]
for i, plot_tuple in enumerate(plot_list):
    plt.subplot(1, 3, i + 1)
    plt.plot(rdsr[plot_tuple[0]], rdsr[plot_tuple[1]], 'o')
    maximum = np.max([rdsr[plot_tuple[0]], rdsr[plot_tuple[1]]]) + 0.02
    plt.axis([1, maximum, 1, maximum])
    plt.xlabel('esr, {}'.format(plot_tuple[0]))
    plt.ylabel('esr, {}'.format(plot_tuple[1]))
    line = np.linspace(1, maximum, 10)
    plt.plot(line, line)
    plt.title('{} vs. {}'.format(plot_tuple[0], plot_tuple[1]))

plt.suptitle('Rise decay steepness ratio')
# plt.show()
filename_figure = 'rdsr_comparison.pdf'
plt.savefig(os.path.join(SAVE_PATH_FIGURES_BAROW, 'sharpness', filename_figure))
