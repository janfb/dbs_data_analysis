import os

import matplotlib.pyplot as plt
import numpy as np

import utils as ut
from definitions import SAVE_PATH_FIGURES_BAROW, SAVE_PATH_DATA_BAROW

"""
read raw data and filter like in the Cole paper, adapted to the data of the barow paper
"""

data_folder = os.path.join(SAVE_PATH_DATA_BAROW, 'raw')
save_folder = os.path.join(SAVE_PATH_DATA_BAROW, 'cleaned')

# read all files in the data folder
file_list = [f for f in os.listdir(data_folder) if f.startswith('subject')]

# for every subject file
for sub, sub_file in enumerate(file_list):
    # load data
    d = ut.load_data_analysis(sub_file, data_folder=data_folder)
    # make new entries in the dict
    d['freqs'] = {}
    d['psd'] = {}

    for i, c in enumerate(d['conditions']):
        # notch filter around artefact in 24-26 Hz subharmonics of 50Hz noise
        d['lfp'][c] = ut.band_pass_filter(d['lfp'][c], d['fs'][c], band=[23, 27], order=3, btype='stop')
        # get psd and save it
        freqs, psd = ut.calculate_psd(d['lfp'][c], d['fs'][c])
        # interpolate between the neighboring bins to remove the dip
        mask = ut.get_array_mask(freqs > 23, freqs < 27)
        psd[mask] = np.mean(psd[[np.where(mask)[0][0]-1, np.where(mask)[0][-1]+1]])
        d['freqs'][c], d['psd'][c] = freqs, psd

        # plotting
        mask = ut.get_array_mask(freqs > 2, freqs < 40)
        plt.plot(d['freqs'][c][mask], d['psd'][c][mask], label=c)
        plt.ylabel('power ')
        plt.xlabel('Frequency [Hz]')

    plt.title('Subject file {}, id {}'.format(sub_file[:-2], d['id']['rest']))
    plt.legend()

    # save data
    ut.save_data(data_dict=d,
                 filename='subject_{}_cleaned_psd.p'.format(d['number']),
                 folder=save_folder)
    plt.savefig(os.path.join(SAVE_PATH_FIGURES_BAROW, 'cleaned', sub_file[:-2] + '_psd.pdf'))
    # plt.show()
    plt.close()
    # break


