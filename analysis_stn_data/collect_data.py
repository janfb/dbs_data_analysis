import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from definitions import SAVE_PATH_FIGURES, SAVE_PATH_DATA, DATA_PATH
import scipy.io
import scipy.signal


data_folder = os.path.join(DATA_PATH, 'STN_data_PAC')
save_folder = os.path.join(DATA_PATH, 'STN_data_PAC', 'collected')
file_list = os.listdir(data_folder)

subject_list = ['DF', 'DP', 'JA', 'JB', 'DS', 'JN', 'JP', 'LM', 'MC', 'MW', 'SW', 'WB']

for subject_id in subject_list:

    subject_file_list = [file for file in file_list if subject_id in file and file.endswith('.mat')]

    # for every subject there should 4 files
    # save them in a dict and save the dict to disk
    subject_dict = dict(lfp={}, pac={}, id=subject_id)

    for file_idx, file in enumerate(subject_file_list):
        subject_dict[file] = scipy.io.loadmat(os.path.join(data_folder, file))

    file_name = 'subject_{}_lfp_and_pac.p'.format(subject_id)
    ut.save_data(subject_dict, file_name, save_folder)