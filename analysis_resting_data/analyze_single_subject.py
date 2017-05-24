import matplotlib.pyplot as plt
import numpy as np
import os
from definitions import SAVE_PATH_FIGURES
from utils import load_data_spm, get_array_mask, band_pass_filter

subject_number = 3
filename = 'spmeeg_{}.mat'.format(subject_number)
channel_name = 'GPiL23'
d = load_data_spm(filename, data_folder='/Users/Jan/Dropbox/Master/LR_Kuehn/data/dystonia_rest/for_python')
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
band = [4, 40]
lfp_filt = band_pass_filter(y=lfp, fs=fs, band=band, plot_response=False)

# construct the time vector
dt = 1 / fs
t = np.arange(0, d['data'].shape[1] * dt, dt)

# design plotting mask
window_length = 30  # in sec
window_start = 60
fontsize = 17
mask = get_array_mask(t > window_start, t < window_start + window_length)
plt.figure(figsize=(12, 4))
plt.plot(t[mask], lfp[mask], label='raw')
plt.plot(t[mask], lfp_filt[mask], label='filtered'.format(band))
plt.ylabel('[$\mu V$]', fontsize=fontsize)
plt.xlabel('time [s]', fontsize=fontsize)
plt.title('lfp, contact pair {}, raw and filtered in band {} Hz'.format(channel_name, band), fontsize=fontsize)
plt.legend(prop={'size': 17})
plt.tight_layout()
# plt.savefig(os.path.join(SAVE_PATH_FIGURES, 'lfp_raw_filtered_example.pdf'))
# plt.show()
plt.close()

t = np.arange(0, 22 * np.pi, 0.01)
x = np.sin(t)
plt.figure(figsize=(12, 4))
plt.plot(t, x)
plt.xticks([], [])
plt.ylim([-1.2, 2])
plt.yticks([])
plt.show()
