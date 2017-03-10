import python.utils as ut
import matplotlib.pyplot as plt
import numpy as np


# load data 
data = ut.load_data_analysis('psd_maxamp_theta_beta.p')

# now make a plot of all selected channels and the histogram over peaks
frequs = data['frequs']

mask = ut.get_array_mask(frequs > 3, frequs < 41)
plt.figure(figsize=(15, 8))
plt.plot(frequs[mask], data['psd_beta'][:, mask].T, linewidth=.7, color='C1', alpha=.5)
plt.plot(frequs[mask], data['psd_theta'][:, mask].T, linewidth=.7, color='C4', alpha=.5)
joined_psd_mean = np.mean(np.vstack((data['psd_theta'], data['psd_beta'])), axis=0)
plt.plot(frequs[mask], joined_psd_mean[mask], label='mean', color='C0')
plt.show()


plt.figure(figsize=(15, 8))
plt.hist(x=data['theta_peaks'])
plt.show()

