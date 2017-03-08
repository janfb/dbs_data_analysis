import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

data_folder = '/Users/Jan/Dropbox/Master/LR_Kuehn/data/dystonia_rest'
filename = 'spmeeg_031IU54.mat'
mat = scipy.io.loadmat(os.path.join(data_folder, filename))

print(mat['D'].shape)