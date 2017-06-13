import os
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
from definitions import SAVE_PATH_FIGURES, SAVE_PATH_DATA, DATA_PATH
import pprint
import scipy.stats
import analysis_stn_data.plotting_functions as plotter


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
                   sig_per_hemi=dict(on=[], off=[]))

esr_results = dict(per_cond_channel=np.zeros((n_bands, n_subject_hemis, n_channels, n_conditions)),
                   per_channel=np.zeros((n_bands, n_subject_hemis, n_channels)),
                   per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                   per_hemi = np.zeros((n_bands, n_subject_hemis)),
                   max_channel_per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                   sig_per_hemi=dict(on=[], off=[]))

rdsr_results = dict(per_cond_channel=np.zeros((n_bands, n_subject_hemis, n_channels, n_conditions)),
                    per_channel=np.zeros((n_bands, n_subject_hemis, n_channels)),
                    per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                    per_hemi = np.zeros((n_bands, n_subject_hemis)),
                    max_channel_per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                    sig_per_hemi=dict(on=[], off=[]))

mpv_results = dict(per_cond_channel=np.zeros((n_bands, n_subject_hemis, n_channels, n_conditions)),
                   per_channel=np.zeros((n_bands, n_subject_hemis, n_channels)),
                   per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                   per_hemi = np.zeros((n_bands, n_subject_hemis)),
                   max_channel_per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                   sig_per_hemi=dict(on=[], off=[]))

beta_amp_results = dict(per_cond_channel=np.zeros((n_bands, n_subject_hemis, n_channels, n_conditions)),
                        per_channel=np.zeros((n_bands, n_subject_hemis, n_channels)),
                        per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                        per_hemi = np.zeros((n_bands, n_subject_hemis)),
                        max_channel_per_condition=np.zeros((n_bands, n_subject_hemis, n_conditions)),
                        sig_per_hemi=dict(on=[], off=[]))

subject_ids = []
sig_subject_ids = []
bands = [[10, 20]]  # depricated but needed for indexing

# for every subject-hemisphere, extract the values into the matrices defined above
for hemi_idx, hemi_dict in enumerate(data_dict.values()):

    subject_id = hemi_dict['id'][:4]
    subject_ids.append(subject_id)

    # prepare ratio matrices
    # shape = (n_channels, n_bands, n_conditions, (mean, std))
    esr_mat = hemi_dict['esr_mat'][:, :, :, 0]  # extract only the mean
    rdsr_mat = hemi_dict['rdsr_mat'][:, :, :, 0]  # extract only the mean
    mpv_mat = hemi_dict['meanPhaseVec_mat'][:, :, :, 0]  # extract only the mean phase vector length
    beta_mat = hemi_dict['beta_amp_mat']
    # now shape = (n_channels, n_bands, n_conditions)

    # for PAC analysis
    # average across low beta and high and across all HFO amplitude frequencies
    # then average across channels
    # or look at channels independently
    # or conditions
    # or both

    condition_bands = hemi_dict['condition_bands']
    conditions = hemi_dict['conditions']
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
        for condition_idx, condition in enumerate(conditions):

            # prelocate a list of results of the current hemisphere-condition
            pac_channel_values = []
            result_channel_values = dict(esr_mat=[], rdsr_mat=[], mpv_mat=[], beta_mat=[])
            result_mat_keys = ['esr_mat', 'rdsr_mat', 'mpv_mat', 'beta_mat']

            for channel_idx, channel_label in enumerate(hemi_dict['channel_labels']):

                # get the band mask for the current condition and channel
                current_band = condition_bands[channel_label][condition_idx]
                phase_band_mask = ut.get_array_mask(f_phase >= current_band[0] , f_phase <= current_band[1]).squeeze()

                # extract pac values for the current phase band, condition and channel
                pac_amp_phase_band = hemi_dict['pac_matrix'][channel_idx, condition_idx, :, phase_band_mask]
                # average over amplitude and selected phase frequencies
                pac_amp_phase_band = pac_amp_phase_band.mean()

                pac_phase_band[channel_idx, condition_idx] = pac_amp_phase_band

                # if the channel is significant, add it to the sig list
                if hemi_dict['significance_flag'][channel_idx, condition_idx]:
                    # append the result for the current significant channel
                    pac_channel_values.append(pac_amp_phase_band)

                # do the same for the other three results types
                if hemi_dict['significance_flag'][channel_idx, condition_idx]:
                    for result_mat_idx, result_mat in enumerate([esr_mat, rdsr_mat, mpv_mat, beta_mat]):
                        result_channel_values[result_mat_keys[result_mat_idx]].append(result_mat[channel_idx, band_idx, condition_idx])

            # append the pac of the channel with the maximum significant pac for later analysis
            if pac_channel_values:
                pac_results['sig_per_hemi'][condition].append(np.max(np.array(pac_channel_values)))
                esr_results['sig_per_hemi'][condition].append(np.max(np.array(result_channel_values['esr_mat'])))
                rdsr_results['sig_per_hemi'][condition].append(np.max(np.array(result_channel_values['rdsr_mat'])))
                mpv_results['sig_per_hemi'][condition].append(np.max(np.array(result_channel_values['mpv_mat'])))
                beta_amp_results['sig_per_hemi'][condition].append(np.max(np.array(result_channel_values['beta_mat'])))
                sig_subject_ids.append(subject_id)

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
            max_beta = beta_mat[max_channel_idx, band_idx, condition_idx]
            pac_results['max_channel_per_condition'][band_idx, hemi_idx, condition_idx] = max_pac
            esr_results['max_channel_per_condition'][band_idx, hemi_idx, condition_idx] = max_esr
            rdsr_results['max_channel_per_condition'][band_idx, hemi_idx, condition_idx] = max_rdsr
            mpv_results['max_channel_per_condition'][band_idx, hemi_idx, condition_idx] = max_mpv
            beta_amp_results['max_channel_per_condition'][band_idx, hemi_idx, condition_idx] = max_beta

        # sharpness analysis: average ratios across channels and conditions
        # or look at channels indipendently or conditions

        # or select the channel with maximum ratio

        # save per condition and channel
        esr_results['per_cond_channel'][band_idx, hemi_idx, :, :] = esr_mat[:, band_idx, :]
        rdsr_results['per_cond_channel'][band_idx, hemi_idx, :, :] = rdsr_mat[:, band_idx, :]
        mpv_results['per_cond_channel'][band_idx, hemi_idx, :, :] = mpv_mat[:, band_idx, :]
        beta_amp_results['per_cond_channel'][band_idx, hemi_idx, :, :] = beta_mat[:, band_idx, :]

        # save esr per channel
        esr_results['per_channel'][band_idx, hemi_idx, :] = esr_mat[:, band_idx, :].mean(axis=-1)  # average over conditions
        rdsr_results['per_channel'][band_idx, hemi_idx, :] = rdsr_mat[:, band_idx, :].mean(axis=-1)
        mpv_results['per_channel'][band_idx, hemi_idx, :] = mpv_mat[:, band_idx, :].mean(axis=-1)
        beta_amp_results['per_channel'][band_idx, hemi_idx, :] = beta_mat[:, band_idx, :].mean(axis=-1)

        # save esr per condition
        esr_results['per_condition'][band_idx, hemi_idx, :] = esr_mat[:, band_idx, :].mean(axis=0)  # average over channels
        rdsr_results['per_condition'][band_idx, hemi_idx, :] = rdsr_mat[:, band_idx, :].mean(axis=0)
        mpv_results['per_condition'][band_idx, hemi_idx, :] = mpv_mat[:, band_idx, :].mean(axis=0)
        beta_amp_results['per_condition'][band_idx, hemi_idx, :] = beta_mat[:, band_idx, :].mean(axis=0)

        # save esr average over channels and conditions
        esr_results['per_hemi'][band_idx, hemi_idx] = esr_mat[:, band_idx, :].mean()
        rdsr_results['per_hemi'][band_idx, hemi_idx] = rdsr_mat[:, band_idx, :].mean()
        mpv_results['per_hemi'][band_idx, hemi_idx] = mpv_mat[:, band_idx, :].mean()
        beta_amp_results['per_hemi'][band_idx, hemi_idx] = beta_mat[:, band_idx, :].mean()

# to simplify plotting for the cases where the data was selected based on significant pac values,
# collapse the different amounts of data in the conditions into a single list / array
for result_dict in [pac_results, esr_results, rdsr_results, mpv_results, beta_amp_results]:
    result_dict['sig_per_hemi']['all'] = result_dict['sig_per_hemi']['on'] + result_dict['sig_per_hemi']['off']

# plt the beta power OFF vs. ON
# plt.figure(figsize=(10, 5))
# beta_per_condition = beta_amp_results['per_condition'].squeeze()
# beta_on = beta_per_condition[:, 1]
# beta_off = beta_per_condition[:, 0]
# plt.bar([0, 1], [beta_off.mean(), beta_on.mean()])
# plt.xticks([0, 1], ['off', 'on'])
# plt.show()

"""
Make a figure for correlations in every channel across subjects and conditions 
subplot bands x channels 
"""

outlier_std_factor = 3

# ratio_strings = ['esr', 'rdsr']
# ratio_matrix_list = [esr_results, rdsr_results]
#
# for ratio_mat_idx, ratio_matrix in enumerate(ratio_matrix_list):
#
#     plot_idx = 0
#     plt.figure(figsize=(12, 8))
#     xlim = [ratio_matrix['per_cond_channel'].flatten().min(), ratio_matrix['per_cond_channel'].flatten().max()]
#     ylim = [pac_results['per_cond_channel'].flatten().min(), pac_results['per_cond_channel'].flatten().max()]
#
#     for band_idx, band in enumerate(bands):
#         for channel_idx in range(n_channels):
#             # take mean over channels and treat hemispheres as samples, combine conditions
#             x_all = ratio_matrix['per_cond_channel'][band_idx, :, channel_idx, :].flatten()
#             y_all = pac_results['per_cond_channel'][band_idx, :, channel_idx, :].flatten()
#             slope, bias, r, p, stderr = scipy.stats.linregress(x_all, y_all)
#             r = round(r, 2)
#             p = round(p, 3)
#
#             plot_idx += 1
#             plt.subplot(n_bands, n_channels, plot_idx)
#             plt.plot(ratio_matrix['per_cond_channel'][band_idx, :, channel_idx, 0].flatten(),
#                      pac_results['per_cond_channel'][band_idx, :, channel_idx, 0].flatten(), '*', markersize=5,
#                      label='off')
#             plt.plot(ratio_matrix['per_cond_channel'][band_idx, :, channel_idx, 1].flatten(),
#                      pac_results['per_cond_channel'][band_idx, :, channel_idx, 1].flatten(), '*', markersize=5,
#                      label='on')
#             plt.xlabel(ratio_strings[ratio_mat_idx])
#
#             if plot_idx == 1 or plot_idx == 4:
#                 plt.ylabel('mean pac')
#
#             # fit a line
#             xvals = np.linspace(x_all.min(), x_all.max(), x_all.size)
#             plt.plot(xvals, bias + slope * xvals, label='r={}, p={}'.format(r, p))
#
#             # plot the regression line with outliers removed
#             x_clean, y_clean, x_out, y_out, mask = ut.exclude_outliers(x_all, y_all, n=outlier_std_factor)  # use n times std away from mean as criterion
#             # plot the outliers
#             outlier_indices = np.where(np.logical_not(mask))[0]
#             outlier_labels = np.repeat(subject_ids, int(x_all.size / len(subject_ids)))
#             for outlier_idx in range(outlier_indices.shape[0]):
#                 plt.plot(x_out[outlier_idx], y_out[outlier_idx], '+', markersize=7,
#                          label=outlier_labels[outlier_indices[outlier_idx]])
#
#             slope, bias, r, p, stderr = scipy.stats.linregress(x_clean, y_clean)
#             xvals = np.linspace(x_clean.min(), x_clean.max(), x_clean.size)
#             plt.plot(xvals, bias + slope * xvals, label='r={}, p={}, cleaned'.format(round(r, 2), round(p, 3)))
#
#             plt.legend(loc=1, prop={'size': 7})
#             plt.title('{}, channel {}'.format(band_str[band_idx], channel_idx + 1))
#             print('across subject and conditions, {}, channel {}'.format(band, channel_idx + 1), r)
#             # plt.xlim(xlim)
#             plt.ylim(ylim)
#
#     plt.suptitle('Correlations between PAC and {} for every channel'.format(ratio_strings[ratio_mat_idx].upper()))
#
#     figure_filename = 'pac_{}_corr_all_channels.pdf'.format(ratio_strings[ratio_mat_idx])
#     plt.savefig(os.path.join(save_folder, figure_filename))
#     plt.show()
#     plt.close()
#
"""
Produce plots of the correlations between PAC and ESR, RDSR and mean phase vector amplitude
    A: pooled over subjects and conditions of the max PAC channels
    B: pooled over subjects, conditions and channels
    C: pooled over subjects and conditions, averaged over channels
"""

# comment in or out to either plot correlations between PAC and MPV or ESR and MPV

y_label = 'mean pac'
x_labels = np.array(['esr', 'rdsr', 'mpv length', 'beta amp']).repeat(3)
x_labels = np.array(['esr', 'rdsr', 'mpv length', 'beta amp']).repeat(3)

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
                   [pac_results['per_condition'], mpv_results['per_condition']],   # average channels
                   [esr_results['per_cond_channel'], beta_amp_results['per_cond_channel']]]

# title_list = ['Correlations between PAC and {}, pooled across conditions, {}'.format(dings.upper(), bumbs)
#               for dings, bumbs in [x_labels, analysis_types]]

title_list = ['Correlations between PAC and ESR, pooled across conditions, max PAC channels',
              'Correlations between PAC and ESR, pooled across conditions and channels',
              'Correlations between PAC and ESR, pooled across conditions, averaged over channels',

              'Correlations between PAC and RDSR, pooled across conditions, max PAC channels',
              'Correlations between PAC and RDSR, pooled across conditions and channels',
              'Correlations between PAC and RDSR, pooled across conditions, averaged over channels',

              'Correlations between PAC and MPV length, pooled across conditions, max PAC channels',
              'Correlations between PAC and MPV length, pooled across conditions and channels',
              'Correlations between PAC and MPV length, pooled across conditions, averaged over channels',

              'Correlations between ESR and beta power, pooled across conditions and channels']

figure_filename_list = ['pac_esr_corr_max_channels.pdf',
                        'pac_esr_corr_pooled.pdf',
                        'pac_esr_corr_average_channels.pdf',

                        'pac_rdsr_corr_max_channels.pdf',
                        'pac_rdsr_corr_pooled.pdf',
                        'pac_rdsr_corr_average_channels.pdf',

                        'pac_mpv_corr_max_channels.pdf',
                        'pac_mpv_corr_pooled.pdf',
                        'pac_mpv_corr_average_channels.pdf',
                        'esr_beta_pooled.pdf']

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

band_idx = 0

# plot figures A, B, C
for data_pair_idx, data_pair in enumerate(data_pairs_list):
    d2, d1 = data_pair

    plt.figure(figsize=(10, 5))
    # ylim = [d2.flatten().min(), d2.flatten().max()]

    # extract the current data
    x_all = d1[band_idx,].flatten()
    y_all = d2[band_idx,].flatten()

    # regress all data points
    slope, bias, r, p, stderr = scipy.stats.linregress(x_all, y_all)
    r = round(r, 2)
    p = round(p, 3)

    # plot all data points, color coded for conditions
    plt.plot(d1[band_idx, ..., 0].flatten(),
             d2[band_idx, ..., 0].flatten(), '*', markersize=5, label='off')
    plt.plot(d1[band_idx, ..., 1].flatten(),
             d2[band_idx, ..., 1].flatten(), '*', markersize=5, label='on')
    plt.xlabel(x_labels[data_pair_idx])

    # only left plot gets ylabel
    if data_pair_idx < 9:
        plt.ylabel(y_label)
    else:
        plt.ylabel('esr')

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
    plt.title('{} n={}'.format(band_str[band_idx], x_all.size))

    plt.suptitle(title_list[data_pair_idx])
    figure_filename = figure_filename_list[data_pair_idx]
    plt.savefig(os.path.join(save_folder, figure_filename))
    # plt.show()
    plt.close()

"""
The significance selected data needs an extra treatment because there can be different amounts of data points in the 
conditions
"""

# plot correlation matrix with beta amp, 5 x 5
n_variables = 5

data_pairs_list = [[pac_results['sig_per_hemi'], esr_results['sig_per_hemi']],
                   [pac_results['sig_per_hemi'], rdsr_results['sig_per_hemi']],
                   [pac_results['sig_per_hemi'], mpv_results['sig_per_hemi']],
                   [pac_results['sig_per_hemi'], beta_amp_results['sig_per_hemi']],
                   [esr_results['sig_per_hemi'], rdsr_results['sig_per_hemi']],
                   [esr_results['sig_per_hemi'], mpv_results['sig_per_hemi']],
                   [esr_results['sig_per_hemi'], beta_amp_results['sig_per_hemi']],
                   [rdsr_results['sig_per_hemi'], mpv_results['sig_per_hemi']],
                   [rdsr_results['sig_per_hemi'], beta_amp_results['sig_per_hemi']],
                   [mpv_results['sig_per_hemi'], beta_amp_results['sig_per_hemi']]]

correlation_matrix_idx_l = [5, 10, 15, 20, 11, 16, 21, 17, 22, 23]
correlation_matrix_idx_u = [1, 2, 3, 4, 7, 8, 9, 13, 14, 19]

title_list1 = [
    'Correlations between PAC and {}, pooled across conditions, max significant PAC channels'.format(dings)
    for dings in ['ESR', 'RDSR', 'MPV', 'beta amp']]
title_list2 = [
    'Correlations between ESR and {}, pooled across conditions, max significant PAC channels'.format(dings)
    for dings in ['RDSR', 'MPV', 'beta amp']]
title_list3 = [
    'Correlations between RDSR and {}, pooled across conditions, max significant PAC channels'.format(dings)
    for dings in ['MPV', 'beta amp']]

title_list4 = [
    'Correlations between PVL and {}, pooled across conditions, max significant PAC channels'.format(dings)
    for dings in ['beta amp']]

title_list = title_list1 + title_list2 + title_list3 + title_list4

figure_filename_list1 = ['pac_{}_corr_max_sig_channels.pdf'.format(dings) for dings in
                         ['esr', 'rdsr', 'pvl', 'beta amp']]
figure_filename_list2 = ['esr_{}_corr_max_sig_channels.pdf'.format(dings) for dings in ['rdsr', 'pvl', 'beta amp']]
figure_filename_list3 = ['rdsr_{}_corr_max_sig_channels.pdf'.format(dings) for dings in ['pvl', 'beta amp']]
figure_filename_list4 = ['pvl_{}_corr_max_sig_channels.pdf'.format(dings) for dings in ['beta amp']]
figure_filename_list = figure_filename_list1 + figure_filename_list2 + figure_filename_list3 + figure_filename_list4

y_labels = np.array(
    ['mean pac', 'mean pac', 'mean pac', 'mean pac', 'esr', 'esr', 'esr', 'rdsr', 'rdsr', 'pvl'])
x_labels = np.array(
    ['esr', 'rdsr', 'pvl', 'beta amp', 'rdsr', 'pvl', 'beta amp', 'pvl', 'beta amp',
     'beta_amp'])

matrix_labels = ['pac', 'esr', 'rdsr', 'pvl', 'beta']

# calculate correlations and collect matrices for this data
corr, biasm, slopem, pm, data = plotter.calculate_sig_channels_and_correlation_matrix(data_pairs_list, x_labels, y_labels,
                                                                                title_list, figure_filename_list,
                                                                                n_bands, sig_subject_ids,
                                                                                outlier_std_factor, n_variables,
                                                                                correlation_matrix_idx_l,
                                                                                correlation_matrix_idx_u,
                                                                                band_str, save_folder)

# plot correlation matrix
plotter.plot_correlation_matrix(corr, matrix_labels, save_folder)

# save it
save_dict = dict(corr=corr, bias_mat=biasm, slope_mat=slopem, p_mat=pm, data=data, variables=matrix_labels)
ut.save_data(save_dict, 'sig_max_channels_correlation_matrices_n{}.p'.format(n_variables), SAVE_PATH_DATA)

# do the same for 4 variables
# plot correlation matrix with pac, esr, rdsr, pvl, 4 x 4
n_variables = 4

data_pairs_list = [[pac_results['sig_per_hemi'], esr_results['sig_per_hemi']],
                   [pac_results['sig_per_hemi'], rdsr_results['sig_per_hemi']],
                   [pac_results['sig_per_hemi'], mpv_results['sig_per_hemi']],
                   [esr_results['sig_per_hemi'], rdsr_results['sig_per_hemi']],
                   [esr_results['sig_per_hemi'], mpv_results['sig_per_hemi']],
                   [rdsr_results['sig_per_hemi'], mpv_results['sig_per_hemi']]]

correlation_matrix_idx_l = [4, 8, 12, 9, 13, 14]
correlation_matrix_idx_u = [1, 2, 3, 6, 7, 11]

title_list1 = [
    'Correlations between PAC and {}, pooled across conditions, max significant PAC channels'.format(dings)
    for dings in ['ESR', 'RDSR', 'MPV']]
title_list2 = [
    'Correlations between ESR and {}, pooled across conditions, max significant PAC channels'.format(dings)
    for dings in ['RDSR', 'MPV']]
title_list3 = [
    'Correlations between RDSR and {}, pooled across conditions, max significant PAC channels'.format(dings)
    for dings in ['MPV']]

title_list = title_list1 + title_list2 + title_list3

figure_filename_list1 = ['pac_{}_corr_max_sig_channels.pdf'.format(dings) for dings in ['esr', 'rdsr', 'mpv']]
figure_filename_list2 = ['esr_{}_corr_max_sig_channels.pdf'.format(dings) for dings in ['rdsr', 'mpv']]
figure_filename_list3 = ['rdsr_{}_corr_max_sig_channels.pdf'.format(dings) for dings in ['mpv']]
figure_filename_list = figure_filename_list1 + figure_filename_list2 + figure_filename_list3

y_labels = np.array(['mean pac', 'mean pac', 'mean pac', 'esr', 'esr', 'rdsr'])
x_labels = np.array(['esr', 'rdsr', 'mpv', 'rdsr', 'mpv', 'mpv'])

matrix_labels = ['pac', 'esr', 'rdsr', 'pvl']

corr, biasm, slopem, pm, data = plotter.calculate_sig_channels_and_correlation_matrix(data_pairs_list, x_labels, y_labels,
                                                                                title_list, figure_filename_list,
                                                                                n_bands, sig_subject_ids,
                                                                                outlier_std_factor, n_variables,
                                                                                correlation_matrix_idx_l,
                                                                                correlation_matrix_idx_u,
                                                                                band_str, save_folder)

plotter.plot_correlation_matrix(corr, matrix_labels, save_folder)

save_dict = dict(corr=corr, bias_mat=biasm, slope_mat=slopem, p_mat=pm, data=data, variables=matrix_labels)
ut.save_data(save_dict, 'sig_max_channels_correlation_matrices_n{}.p'.format(n_variables), os.path.join(SAVE_PATH_DATA, 'stn'))
print('p-vals:')
print(pm)