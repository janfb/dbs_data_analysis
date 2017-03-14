import os
import numpy as np
import matplotlib.pyplot as plt
from definitions import SAVE_PATH_FIGURES, SAVE_PATH_DATA, DATA_PATH
from python.utils import load_data_spm, remove_50_noise, calculate_psd, find_peak_in_band, get_array_mask, save_data, band_pass_filter

filename = 'spmeeg_19.mat'
channel_name = 'GPiR23'
d = load_data_spm(filename)
fs = d['fsample'][0][0]

lfp = None
for idx, str in enumerate(d['chanlabels'][0]):
    if str[0] == channel_name:
        lfp = d['data'][idx]

if lfp is None:
    raise ValueError('Could not find the specified channel in the data file: {}'.format(channel_name))

# remove mean
lfp = lfp - np.mean(lfp)

# filter in theta range:
lfp_filt = band_pass_filter(y=lfp, fs=fs, band=[4, 12], order=4)

# construct the time vector
dt = 1 / fs
t = np.arange(0, d['data'].shape[1] * dt, dt)

# design plotting mask
window_length = 3  # in sec
window_start = 250
mask = get_array_mask(t > window_start, t < window_start + window_length)
plt.figure(figsize=(7, 3))
plt.plot(t[mask], lfp[mask], label='raw')
plt.plot(t[mask], lfp_filt[mask], label='filtered')
plt.ylabel('[$\muV]')
plt.xlabel('time [s]')
plt.title('contact pair GPiR23 from case 19')
plt.legend()
plt.savefig(os.path.join(SAVE_PATH_FIGURES, 'figure_1C.pdf'))
plt.show()



