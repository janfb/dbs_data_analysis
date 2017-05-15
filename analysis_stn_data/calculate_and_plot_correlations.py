import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from definitions import SAVE_PATH_FIGURES, SAVE_PATH_DATA, DATA_PATH
import pprint
import scipy.stats

data_folder = os.path.join(SAVE_PATH_DATA, 'stn')
save_folder = os.path.join(SAVE_PATH_FIGURES, 'stn', 'optimized_bands')
filename = 'sharpness_pac_separated_in_hemisphere_n12_optimized_bands.p'

data_dict = ut.load_data_analysis(filename, data_folder)
n_subject_hemis = len(data_dict.keys())
n_channels = 3
n_bands = 1
n_conditions = 2
band_str = ['optimized band']
file_addon = '_optimized_bands'

pac_results = dict(per_cond_channel=np.zeros((n_bands, n_subject_hemis, n_channels, n_conditions)),
                   per_channel=np.zeros((n_bands, n_subject_hemis, n_channels)),
                   per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                   per_hemi = np.zeros((n_bands, n_subject_hemis)),
                   max_channel_per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                   sig_per_hemi=np.zeros((n_bands, n_subject_hemis)))

esr_results = dict(per_cond_channel=np.zeros((n_bands, n_subject_hemis, n_channels, n_conditions)),
                   per_channel=np.zeros((n_bands, n_subject_hemis, n_channels)),
                   per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                   per_hemi = np.zeros((n_bands, n_subject_hemis)),
                   max_channel_per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                   sig_per_hemi=np.zeros((n_bands, n_subject_hemis)))

rdsr_results = dict(per_cond_channel=np.zeros((n_bands, n_subject_hemis, n_channels, n_conditions)),
                    per_channel=np.zeros((n_bands, n_subject_hemis, n_channels)),
                    per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                    per_hemi = np.zeros((n_bands, n_subject_hemis)),
                    max_channel_per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                    sig_per_hemi=np.zeros((n_bands, n_subject_hemis)))

mpv_results = dict(per_cond_channel=np.zeros((n_bands, n_subject_hemis, n_channels, n_conditions)),
                   per_channel=np.zeros((n_bands, n_subject_hemis, n_channels)),
                   per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                   per_hemi = np.zeros((n_bands, n_subject_hemis)),
                   max_channel_per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                   sig_per_hemi=np.zeros((n_bands, n_subject_hemis)))

subject_ids = []
bands = [[10, 20]]  # depricated but needed for indexing

# for every subject-hemisphere, extract the values into the matrices defined above
for hemi_idx, hemi_dict in enumerate(data_dict.values()):

    subject_ids.append(hemi_dict['id'][:4])

    # prepare ratio matrices
    # shape = (n_channels, n_bands, n_conditions, (mean, std))
    esr_mat = hemi_dict['esr_mat'][:, :, :, 0]  # extract only the mean
    rdsr_mat = hemi_dict['rdsr_mat'][:, :, :, 0]  # extract only the mean
    mpv_mat = hemi_dict['meanPhaseVec_mat'][:, :, :, 0]  # extract only the mean phase vector length
    # now shape = (n_channels, n_bands, n_conditions)

    # for PAC analysis
    # average across low beta and high and across all HFO amplitude frequencies
    # then average across channels
    # or look at channels independently
    # or conditions
    # or both

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

        # the bands depends on the condition and on the channel for optimized bands
        pac_phase_band = np.zeros((n_channels, n_conditions))
        for channel_idx, channel_label in enumerate(hemi_dict['channel_labels']):
            for condition_idx in range(n_conditions):

                # get the band mask for the current condition and channel
                current_band = condition_bands[channel_label[0]][condition_idx]
                phase_band_mask = ut.get_array_mask(f_phase >= current_band[0] , f_phase <= current_band[1]).squeeze()

                # extract pac values for the current phase band, condition and channel
                pac_amp_phase_band = hemi_dict['pac_matrix'][channel_idx, condition_idx, :, phase_band_mask]
                # average over amplitude and selected phase frequencies
                pac_amp_phase_band = pac_amp_phase_band.mean()

                pac_phase_band[channel_idx, condition_idx] = pac_amp_phase_band

        # save per condition and channel
        pac_results['per_cond_channel'][band_idx, hemi_idx, :, :] = pac_phase_band

        # save pac per channel
        pac_results['per_channel'][band_idx, hemi_idx, :] = pac_phase_band.mean(axis=1)  # average over conditions

        # save per condition
        pac_results['per_condition'][band_idx, hemi_idx, :] = pac_phase_band.mean(axis=0)  # average over channels

        # save pac average over channels and conditions
        pac_results['per_hemi'][band_idx, hemi_idx] = pac_phase_band.mean()

        # save the values of the channel with maximum PAC for every hemi and condition
        for condition_idx in range(n_conditions):
            # get the idx of the maximum channel for the current condition
            max_channel_idx = pac_phase_band[:, condition_idx].argmax()
            # get the corresponding pac value
            max_pac = pac_phase_band[max_channel_idx, condition_idx]
            # and the ratio values
            max_esr = esr_mat[max_channel_idx, band_idx, condition_idx]
            max_rdsr = rdsr_mat[max_channel_idx, band_idx, condition_idx]
            max_mpv = mpv_mat[max_channel_idx, band_idx, condition_idx]
            pac_results['max_channel_per_condition'][band_idx, hemi_idx, condition_idx] = max_pac
            esr_results['max_channel_per_condition'][band_idx, hemi_idx, condition_idx] = max_esr
            rdsr_results['max_channel_per_condition'][band_idx, hemi_idx, condition_idx] = max_rdsr
            mpv_results['max_channel_per_condition'][band_idx, hemi_idx, condition_idx] = max_mpv

        # sharpness analysis: average ratios across channels and conditions
        # or look at channels indipendently or conditions

        # or select the channel with maximum ratio

        # save per condition and channel
        esr_results['per_cond_channel'][band_idx, hemi_idx, :, :] = esr_mat[:, band_idx, :]
        rdsr_results['per_cond_channel'][band_idx, hemi_idx, :, :] = rdsr_mat[:, band_idx, :]
        mpv_results['per_cond_channel'][band_idx, hemi_idx, :, :] = mpv_mat[:, band_idx, :]

        # save esr per channel
        esr_results['per_channel'][band_idx, hemi_idx, :] = esr_mat[:, band_idx, :].mean(axis=-1)  # average over conditions
        rdsr_results['per_channel'][band_idx, hemi_idx, :] = rdsr_mat[:, band_idx, :].mean(axis=-1)
        mpv_results['per_channel'][band_idx, hemi_idx, :] = mpv_mat[:, band_idx, :].mean(axis=-1)

        # save esr per condition
        esr_results['per_condition'][band_idx, hemi_idx, :] = esr_mat[:, band_idx, :].mean(axis=0)  # average over channels
        rdsr_results['per_condition'][band_idx, hemi_idx, :] = rdsr_mat[:, band_idx, :].mean(axis=0)
        mpv_results['per_condition'][band_idx, hemi_idx, :] = mpv_mat[:, band_idx, :].mean(axis=0)

        # save esr average over channels and conditions
        esr_results['per_hemi'][band_idx, hemi_idx] = esr_mat[:, band_idx, :].mean()
        rdsr_results['per_hemi'][band_idx, hemi_idx] = rdsr_mat[:, band_idx, :].mean()
        mpv_results['per_hemi'][band_idx, hemi_idx] = mpv_mat[:, band_idx, :].mean()


"""
Make a figure for correlations in every channel across subjects and conditions 
subplot bands x channels 
"""

outlier_std_factor = 4

ratio_strings = ['esr', 'rdsr']
ratio_matrix_list = [esr_results, rdsr_results]

for ratio_mat_idx, ratio_matrix in enumerate(ratio_matrix_list):

    plot_idx = 0
    plt.figure(figsize=(12, 8))
    xlim = [ratio_matrix['per_cond_channel'].flatten().min(), ratio_matrix['per_cond_channel'].flatten().max()]
    ylim = [pac_results['per_cond_channel'].flatten().min(), pac_results['per_cond_channel'].flatten().max()]

    for band_idx, band in enumerate(bands):
        for channel_idx in range(n_channels):
            # take mean over channels and treat hemispheres as samples, combine conditions
            x_all = ratio_matrix['per_cond_channel'][band_idx, :, channel_idx, :].flatten()
            y_all = pac_results['per_cond_channel'][band_idx, :, channel_idx, :].flatten()
            slope, bias, r, p, stderr = scipy.stats.linregress(x_all, y_all)
            r = round(r, 2)
            p = round(p, 3)

            plot_idx += 1
            plt.subplot(n_bands, n_channels, plot_idx)
            plt.plot(ratio_matrix['per_cond_channel'][band_idx, :, channel_idx, 0].flatten(),
                     pac_results['per_cond_channel'][band_idx, :, channel_idx, 0].flatten(), '*', markersize=5,
                     label='off')
            plt.plot(ratio_matrix['per_cond_channel'][band_idx, :, channel_idx, 1].flatten(),
                     pac_results['per_cond_channel'][band_idx, :, channel_idx, 1].flatten(), '*', markersize=5,
                     label='on')
            plt.xlabel(ratio_strings[ratio_mat_idx])

            if plot_idx == 1 or plot_idx == 4:
                plt.ylabel('mean pac')

            # fit a line
            xvals = np.linspace(x_all.min(), x_all.max(), x_all.size)
            plt.plot(xvals, bias + slope * xvals, label='r={}, p={}'.format(r, p))

            # plot the regression line with outliers removed
            x_clean, y_clean, x_out, y_out, mask = ut.exclude_outliers(x_all, y_all, n=outlier_std_factor)  # use n times std away from mean as criterion
            # plot the outliers
            outlier_indices = np.where(np.logical_not(mask))[0]
            outlier_labels = np.repeat(subject_ids, int(x_all.size / len(subject_ids)))
            for outlier_idx in range(outlier_indices.shape[0]):
                plt.plot(x_out[outlier_idx], y_out[outlier_idx], '+', markersize=7,
                         label=outlier_labels[outlier_indices[outlier_idx]])

            slope, bias, r, p, stderr = scipy.stats.linregress(x_clean, y_clean)
            xvals = np.linspace(x_clean.min(), x_clean.max(), x_clean.size)
            plt.plot(xvals, bias + slope * xvals, label='r={}, p={}, cleaned'.format(round(r, 2), round(p, 3)))

            plt.legend(loc=1, prop={'size': 7})
            plt.title('{}, channel {}'.format(band_str[band_idx], channel_idx + 1))
            print('across subject and conditions, {}, channel {}'.format(band, channel_idx + 1), r)
            # plt.xlim(xlim)
            plt.ylim(ylim)

    plt.suptitle('Correlations between PAC and {} for every channel'.format(ratio_strings[ratio_mat_idx].upper()))

    figure_filename = 'pac_{}_corr_all_channels.pdf'.format(ratio_strings[ratio_mat_idx])
    plt.savefig(os.path.join(save_folder, figure_filename))
    plt.show()
    plt.close()

"""
Produce plots of the correlations between PAC and ESR, RDSR and mean phase vector amplitude 
    A: pooled over subjects and conditions of the max PAC channels 
    B: pooled over subjects, conditions and channels 
    C: pooled over subjects and conditions, averaged over channels 
"""

# comment in or out to either plot correlations between PAC and MPV or ESR and MPV

# make a list of data series pairs to have only a single for loop for the figure
data_pairs_list = [  # first pac vs. esr
                   [pac_results['max_channel_per_condition'], esr_results['max_channel_per_condition']],
                   [pac_results['per_cond_channel'], esr_results['per_cond_channel']],  # all pooled
                   [pac_results['per_condition'], esr_results['per_condition']],
                    # then pac vs. rdsr
                   [pac_results['max_channel_per_condition'], rdsr_results['max_channel_per_condition']],
                   [pac_results['per_cond_channel'], rdsr_results['per_cond_channel']],  # all pooled
                   [pac_results['per_condition'], rdsr_results['per_condition']],
                    # and finally pac vs. mpv
                   [pac_results['max_channel_per_condition'], mpv_results['max_channel_per_condition']],  # A over max channels
                   [pac_results['per_cond_channel'], mpv_results['per_cond_channel']],  # all pooled
                   [pac_results['per_condition'], mpv_results['per_condition']]]  # average channels

title_list = ['Correlations between PAC and ESR, pooled across conditions, max PAC channels',
              'Correlations between PAC and ESR, pooled across conditions and channels',
              'Correlations between PAC and ESR, pooled across conditions, averaged over channels',

              'Correlations between PAC and RDSR, pooled across conditions, max PAC channels',
              'Correlations between PAC and RDSR, pooled across conditions and channels',
              'Correlations between PAC and RDSR, pooled across conditions, averaged over channels',

              'Correlations between PAC and MPV length, pooled across conditions, max PAC channels',
              'Correlations between PAC and MPV length, pooled across conditions and channels',
              'Correlations between PAC and MPV length, pooled across conditions, averaged over channels']

figure_filename_list = ['pac_esr_corr_max_channels.pdf',
                        'pac_esr_corr_pooled.pdf',
                        'pac_esr_corr_average_channels.pdf',

                        'pac_rdsr_corr_max_channels.pdf',
                        'pac_rdsr_corr_pooled.pdf',
                        'pac_rdsr_corr_average_channels.pdf',

                        'pac_mpv_corr_max_channels.pdf',
                        'pac_mpv_corr_pooled.pdf',
                        'pac_mpv_corr_average_channels.pdf']
y_label = 'mean pac'
x_labels = np.array(['esr', 'rdsr', 'mpv length']).repeat(3)


# same between mpv and ratio
# data_pairs_list = [[mpv_results['per_cond_channel'], ratio_matrix['per_cond_channel']],  # all pooled
#                    [mpv_results['per_condition'], ratio_matrix['per_condition']]]  # average channels
#
# title_list = ['Correlations between {} and MPV length, pooled across conditions and channels'.format(ratio_str.upper()),
#               'Correlations between {} and MPV length, pooled across conditions, averaged over channels'.format(ratio_str.upper())]
#
# figure_filename_list = ['{}_mpv_corr_pooled.pdf'.format(ratio_str),
#                         '{}_mpv_corr_average_channels.pdf'.format(ratio_str)]
#
# y_label = ratio_str

# plot figures A, B, C
for data_pair_idx, data_pair in enumerate(data_pairs_list):
    plot_idx = 0
    d2, d1 = data_pair

    plt.figure(figsize=(10, 5))
    # ylim = [d2.flatten().min(), d2.flatten().max()]

    for band_idx, band in enumerate(bands):

        # extract the current data
        x_all = d1[band_idx,].flatten()
        y_all = d2[band_idx,].flatten()

        # regress all data points
        slope, bias, r, p, stderr = scipy.stats.linregress(x_all, y_all)
        r = round(r, 2)
        p = round(p, 3)

        # plot all data points, color coded for conditions
        plot_idx += 1
        plt.subplot(1, n_bands, plot_idx)
        plt.plot(d1[band_idx, ..., 0].flatten(),
                 d2[band_idx, ..., 0].flatten(), '*', markersize=5, label='off')
        plt.plot(d1[band_idx, ..., 1].flatten(),
                 d2[band_idx, ..., 1].flatten(), '*', markersize=5, label='on')
        plt.xlabel(x_labels[data_pair_idx])

        # only left plot gets ylabel
        if plot_idx == 1:
         plt.ylabel(y_label)

        # plot two correlation line, one for all points, one for selected points (outlier free)
        # plot all
        xvals = np.linspace(x_all.min(), x_all.max(), x_all.size)
        plt.plot(xvals, bias + slope * xvals, label='r={}, p={}'.format(r, p))

        # plot the regression line with outliers removed
        # use n times std away from mean as criterion
        x_clean, y_clean, x_out, y_out, mask = ut.exclude_outliers(x_all, y_all,
                                                               n=outlier_std_factor)

        # plot the outliers
        outlier_indices = np.where(np.logical_not(mask))[0]
        outlier_labels = np.repeat(subject_ids, int(x_all.size / len(subject_ids)))
        for outlier_idx in range(outlier_indices.shape[0]):
            plt.plot(x_out[outlier_idx], y_out[outlier_idx], '+', markersize=7,
                     label=outlier_labels[outlier_indices[outlier_idx]])

        # plot the new regression line
        slope, bias, r, p, stderr = scipy.stats.linregress(x_clean, y_clean)
        xvals = np.linspace(x_clean.min(), x_clean.max(), x_clean.size)
        plt.plot(xvals, bias + slope * xvals, label='r={}, p={}, cleaned'.format(round(r, 2), round(p, 4)))

        plt.legend(loc=1, prop={'size': 7})
        # plt.ylim(ylim)
        plt.title('{}'.format(band_str[band_idx]))

    plt.suptitle(title_list[data_pair_idx])
    figure_filename = figure_filename_list[data_pair_idx]
    plt.savefig(os.path.join(save_folder, figure_filename))
    plt.show()
    plt.close()
