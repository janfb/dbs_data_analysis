import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from definitions import SAVE_PATH_FIGURES, SAVE_PATH_DATA, DATA_PATH
import pprint
import scipy.stats

data_folder = os.path.join(SAVE_PATH_DATA, 'stn')
save_folder = os.path.join(SAVE_PATH_FIGURES, 'stn')
filename = 'sharpness_pac_separated_in_hemisphere_n12_optimized_bands.p'

data_dict = ut.load_data_analysis(filename, data_folder)
n_subject_hemis = len(data_dict.keys())
n_channels = 3
n_bands = 2
n_conditions = 2
band_str = ['optimized band', 'high beta']
file_addon = '_optimized_bands'

pac_results = dict(per_cond_channel=np.zeros((n_bands, n_subject_hemis, n_channels, n_conditions)),
                   per_channel=np.zeros((n_bands, n_subject_hemis, n_channels)),
                   per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                   per_hemi = np.zeros((n_bands, n_subject_hemis)),
                   max_channel_per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)))

esr_results = dict(per_cond_channel=np.zeros((n_bands, n_subject_hemis, n_channels, n_conditions)),
                   per_channel=np.zeros((n_bands, n_subject_hemis, n_channels)),
                   per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                   per_hemi = np.zeros((n_bands, n_subject_hemis)),
                   max_channel_per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)))

rdsr_results = dict(per_cond_channel=np.zeros((n_bands, n_subject_hemis, n_channels, n_conditions)),
                    per_channel=np.zeros((n_bands, n_subject_hemis, n_channels)),
                    per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                    per_hemi = np.zeros((n_bands, n_subject_hemis)),
                    max_channel_per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)))

# for every subject-hemisphere, extract the values into the matrices defined above
for hemi_idx, hemi_dict in enumerate(data_dict.values()):

    id = hemi_dict['id']

    # prepare ratio matrices
    # shape = (n_channels, n_bands, n_conditions, (mean, std))
    esr_mat = hemi_dict['esr_mat'][:, :, :, 0]  # extract only the mean
    rdsr_mat = hemi_dict['rdsr_mat'][:, :, :, 0]  # extract only the mean
    # now shape = (n_channels, n_bands, n_conditions)


    # for PAC analysis
    # average across low beta and high and across all HFO amplitude frequencies
    # then average across channels
    # or look at channels independently
    # or conditions
    # or both

    bands = [[10, 20]]  # depricated but needed for indexing
    condition_bands = hemi_dict['condition_bands']
    f_phase = hemi_dict['f_phase']

    # do it for every band
    for band_idx, band in enumerate(bands):
        # for PAC analysis
        # average across low beta and high and across all HFO amplitude frequencies
        # then average across channels
        # or look at channels independently
        # or conditions
        # or both

        # the bands depends on the condition for optimized bands
        pac_phase_band = np.zeros((n_channels, n_conditions))
        for condition_idx in range(n_conditions):

            # get the mask for the current band
            current_band = condition_bands[condition_idx]
            phase_band_mask = ut.get_array_mask(f_phase >= current_band[0] , f_phase <= current_band[1]).squeeze()

            # extract pac values for the current phase band and condition
            pac_amp_phase_band = hemi_dict['pac_matrix'][:, condition_idx, :, phase_band_mask]
            # average over amplitude and selected phase frequencies
            pac_amp_phase_band = pac_amp_phase_band.mean(axis=(0, 2))

            pac_phase_band[:, condition_idx] = pac_amp_phase_band

        # save per condition and channel
        pac_results['per_cond_channel'][band_idx, hemi_idx, :, :] = pac_phase_band

        # save pac per channel
        pac_results['per_channel'][band_idx, hemi_idx, :] = pac_phase_band.mean(axis=1)  # average over conditions

        # save per condition
        pac_results['per_condition'][band_idx, hemi_idx, :] = pac_phase_band.mean(axis=0)  # average over channels

        # save pac average over channels and conditions
        pac_results['per_hemi'][band_idx, hemi_idx] = pac_phase_band.mean()

        for condition_idx in range(n_conditions):
            # save the values of the channel with maximum PAC for every hemi and condition
            # get the idx of the maximum channel for the current condition
            max_channel_idx = pac_phase_band[:, condition_idx].argmax()
            # get the corresponding pac value
            max_pac = pac_phase_band[max_channel_idx, condition_idx]
            # and the ratio values
            max_esr = esr_mat[max_channel_idx, band_idx, condition_idx]
            max_rdsr = rdsr_mat[max_channel_idx, band_idx, condition_idx]
            pac_results['max_channel_per_condition'][band_idx, hemi_idx, condition_idx] = max_pac
            esr_results['max_channel_per_condition'][band_idx, hemi_idx, condition_idx] = max_esr
            rdsr_results['max_channel_per_condition'][band_idx, hemi_idx, condition_idx] = max_rdsr

        # sharpness analysis: average ratios across channels and conditions
        # or look at channels indipendently or conditions

        # or select the channel with maximum ratio

        # save per condition and channel
        esr_results['per_cond_channel'][band_idx, hemi_idx, :, :] = esr_mat[:, band_idx, :]
        rdsr_results['per_cond_channel'][band_idx, hemi_idx, :, :] = rdsr_mat[:, band_idx, :]

        # save esr per channel
        esr_results['per_channel'][band_idx, hemi_idx, :] = esr_mat[:, band_idx, :].mean(axis=-1)  # average over conditions
        rdsr_results['per_channel'][band_idx, hemi_idx, :] = rdsr_mat[:, band_idx, :].mean(axis=-1)

        # save esr per condition
        esr_results['per_condition'][band_idx, hemi_idx, :] = esr_mat[:, band_idx, :].mean(axis=0)  # average over channels
        rdsr_results['per_condition'][band_idx, hemi_idx, :] = rdsr_mat[:, band_idx, :].mean(axis=0)

        # save esr average over channels and conditions
        esr_results['per_hemi'][band_idx, hemi_idx] = esr_mat[:, band_idx, :].mean()
        rdsr_results['per_hemi'][band_idx, hemi_idx] = rdsr_mat[:, band_idx, :].mean()


"""
Make a figure for correlations in every channel across subjects and conditions 
subplot bands x channels 
"""

plot_idx = 0
plt.figure(figsize=(12, 8))

# CHOOSE BETWEEN ESR OR RDSR DATA HERE
ratio_matrix = esr_results
ratio_str = 'esr'
outlier_std_factor = 4

for band_idx, band in enumerate(bands):
    for channel_idx in range(n_channels):
        # take mean over channels and treat hemispheres as samples, combine conditions
        x = ratio_matrix['per_cond_channel'][band_idx, :, channel_idx, :].flatten()
        y = pac_results['per_cond_channel'][band_idx, :, channel_idx, :].flatten()
        slope, bias, r, p, stderr = scipy.stats.linregress(x, y)
        r = round(r, 2)
        p = round(p, 3)

        plot_idx += 1
        plt.subplot(n_bands, n_channels, plot_idx)
        plt.plot(ratio_matrix['per_cond_channel'][band_idx, :, channel_idx, 0].flatten(),
                 pac_results['per_cond_channel'][band_idx, :, channel_idx, 0].flatten(), '*', markersize=5, label='off')
        plt.plot(ratio_matrix['per_cond_channel'][band_idx, :, channel_idx, 1].flatten(),
                 pac_results['per_cond_channel'][band_idx, :, channel_idx, 1].flatten(), '*', markersize=5, label='on')
        if plot_idx > 3:
            plt.xlabel(ratio_str)
        # else:
        #     plt.xticks([], [])

        if plot_idx == 1 or plot_idx == 4:
            plt.ylabel('mean pac')
        else:
            plt.yticks([], [])

        # fit a line
        xvals = np.linspace(x.min(), x.max(), x.size)
        plt.plot(xvals, bias + slope * xvals, label='r={}, p={}'.format(r, p))

        # plot the regression line with outliers removed
        x, y, x_out, y_out = ut.exclude_outliers(x, y, n=outlier_std_factor)  # use n times std away from mean as criterion
        # plot the outliers
        plt.plot(x_out, y_out, '+', color='C3', markersize=7)

        slope, bias, r, p, stderr = scipy.stats.linregress(x, y)
        xvals = np.linspace(x.min(), x.max(), x.size)
        plt.plot(xvals, bias + slope * xvals, label='r={}, p={}, cleaned'.format(round(r, 2), round(p, 3)))

        plt.legend(loc=1, prop={'size': 7})
        plt.ylim([0.005, 0.035])
        plt.title('{}, channel {}'.format(band_str[band_idx], channel_idx + 1))
        # plt.xlim([1.05, 1.25])
        print('across subject and conditions, {}, channel {}'.format(band, channel_idx + 1), r)

plt.suptitle('Correlations between PAC and {} for every channel'.format(ratio_str.upper()))


figure_filename = 'pac_{}_corr_all_channels{}.pdf'.format(ratio_str, file_addon)
plt.savefig(os.path.join(save_folder, figure_filename))
plt.show()
plt.close()

"""
Make a similar figure with correlations of selected channels with maximum PAC 
"""

plot_idx = 0
plt.figure(figsize=(10, 5))

for band_idx, band in enumerate(bands):
    x = ratio_matrix['max_channel_per_condition'][band_idx, ].flatten()
    y = pac_results['max_channel_per_condition'][band_idx,].flatten()
    slope, bias, r, p, stderr = scipy.stats.linregress(x, y)
    r = round(r, 2)
    p = round(p, 3)

    plot_idx += 1
    plt.subplot(1, n_bands, plot_idx)
    plt.plot(ratio_matrix['max_channel_per_condition'][band_idx, :, 0].flatten(),
             pac_results['max_channel_per_condition'][band_idx, :, 0].flatten(), '*', markersize=5, label='off')
    plt.plot(ratio_matrix['max_channel_per_condition'][band_idx, :, 1].flatten(),
             pac_results['max_channel_per_condition'][band_idx, :, 1].flatten(), '*', markersize=5, label='on')
    plt.xlabel(ratio_str)

    if plot_idx == 1:
        plt.ylabel('mean pac')

    # fit a line
    xvals = np.linspace(x.min(), x.max(), x.size)
    plt.plot(xvals, bias + slope * xvals, label='r={}, p={}'.format(r, p))

    # plot the regression line with outliers removed
    x, y, x_out, y_out = ut.exclude_outliers(x, y, n=outlier_std_factor)  # use n times std away from mean as criterion
    # plot the outliers
    plt.plot(x_out, y_out, '+', color='C3', markersize=7)

    # plot the new regression line
    slope, bias, r, p, stderr = scipy.stats.linregress(x, y)
    xvals = np.linspace(x.min(), x.max(), x.size)
    plt.plot(xvals, bias + slope * xvals, label='r={}, p={}, cleaned'.format(round(r, 2), round(p, 3)))

    plt.legend(loc=1, prop={'size': 7})
    plt.ylim([0.006, 0.036])
    plt.title('{}'.format(band_str[band_idx]))

    print('across subject and conditions, {}, selected channels'.format(band), r)

plt.suptitle('Correlations between PAC and {} for maximum PAC channels'.format(ratio_str.upper()))
figure_filename = 'pac_{}_corr_max_channels{}.pdf'.format(ratio_str, file_addon)
plt.savefig(os.path.join(save_folder, figure_filename))
plt.show()
plt.close()

"""
Pool data across subjects, channels and conditions 
"""

plot_idx = 0
plt.figure(figsize=(10, 5))

for band_idx, band in enumerate(bands):
    x = ratio_matrix['per_cond_channel'][band_idx, ].flatten()
    y = pac_results['per_cond_channel'][band_idx, ].flatten()
    slope, bias, r, p, stderr = scipy.stats.linregress(x, y)
    r = round(r, 2)
    p = round(p, 3)

    plot_idx += 1
    plt.subplot(1, n_bands, plot_idx)
    plt.plot(ratio_matrix['per_cond_channel'][band_idx, :, :, 0].flatten(),
             pac_results['per_cond_channel'][band_idx, :, :, 0].flatten(), '*', markersize=5, label='off')
    plt.plot(ratio_matrix['per_cond_channel'][band_idx, :, :, 1].flatten(),
             pac_results['per_cond_channel'][band_idx, :, :, 1].flatten(), '*', markersize=5, label='on')
    plt.xlabel(ratio_str)

    if plot_idx == 1:
        plt.ylabel('mean pac')

    # fit a line
    xvals = np.linspace(x.min(), x.max(), x.size)
    plt.plot(xvals, bias + slope * xvals, label='r={}, p={}'.format(r, p))

    # plot the regression line with outliers removed
    x, y, x_out, y_out = ut.exclude_outliers(x, y, n=outlier_std_factor)  # use n times std away from mean as criterion
    # plot the outliers
    plt.plot(x_out, y_out, '+', color='C3', markersize=7)

    # plot the new regression line
    slope, bias, r, p, stderr = scipy.stats.linregress(x, y)
    xvals = np.linspace(x.min(), x.max(), x.size)
    plt.plot(xvals, bias + slope * xvals, label='r={}, p={}, cleaned'.format(round(r, 2), round(p, 3)))

    plt.legend(loc=1, prop={'size': 7})
    plt.ylim([0.006, 0.036])
    plt.title('{}'.format(band_str[band_idx]))

    print('across subject, channels and conditions, {}'.format(band), r)

plt.suptitle('Correlations between PAC and {} pooled across channels and conditions'.format(ratio_str.upper()))
figure_filename = 'pac_{}_corr_pooled{}.pdf'.format(ratio_str, file_addon)
plt.savefig(os.path.join(save_folder, figure_filename))
plt.show()
plt.close()
