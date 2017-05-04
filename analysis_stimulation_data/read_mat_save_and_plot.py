import os

import matplotlib.pyplot as plt
import numpy as np

import utils as ut
from definitions import SAVE_PATH_FIGURES_BAROW, DATA_PATH_BAROW, SAVE_PATH_DATA_BAROW

file_list = os.listdir(DATA_PATH_BAROW)
save_folder = os.path.join(SAVE_PATH_DATA_BAROW, 'raw')

n_subjects = 16
suffix = ''

for sub in np.arange(1, n_subjects + 1):

    # save all data of current subject in dict
    subject_dict = dict(filenames={}, lfp={}, fs={}, conditions={}, chanlabels={}, time={}, id={})
    plt.figure(figsize=(7, 5))
    plot_idx = 1
    for f in file_list:
        # process only the current subject
        if f.startswith('spmeeg_{}_'.format(sub)) and not f.startswith('spmeeg_{}_d'.format(sub)):
            current_file_name = f[:-4]
            print('plotting from {}'.format(f))
            d = ut.load_data_spm(f, data_folder=DATA_PATH_BAROW)
            fs = d['fsample'][0][0]
            chanlabels = d['chanlabels'][0][0]
            condition = d['condition_str'][0]

            # save data
            subject_dict['lfp'][condition] = d['data'][0, :]
            subject_dict['filenames'][condition] = current_file_name
            subject_dict['fs'][condition] = fs
            subject_dict['conditions'][condition] = condition
            subject_dict['id'][condition] = d['subject_id'][0]
            subject_dict['chanlabels'][condition] = chanlabels[0]
            subject_dict['number'] = sub

            # downsample to make plotting faster
            down_factor = 300
            time = d['time'][0, ::down_factor]
            lfp = d['data'][0, ::down_factor]

            plt.subplot(3, 1, plot_idx)
            plt.plot(time, lfp)
            plt.ylabel('lfp [mu V]')
            plt.title('Channel {} in condition {}'.format(chanlabels[0], condition))
            if plot_idx < 3:
                plt.xticks([], [])
            else:
                plt.xlabel('Time [s]')
            plot_idx += 1
            plt.suptitle('Subject file {}, id {}'.format(current_file_name, d['subject_id'][0]))

    # save data
    ut.save_data(data_dict=subject_dict,
                 filename='subject_{}_allconditions_raw.p'.format(sub),
                 folder=save_folder)
    plt.savefig(os.path.join(SAVE_PATH_FIGURES_BAROW, 'raw', current_file_name + '_raw.pdf'.format(suffix)))
    # plt.show()
    plt.close()
    # break



