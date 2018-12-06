import numpy as np
import matplotlib.pyplot as plt
import os
from definitions import SAVE_PATH_FIGURES
import utils as ut
import scipy.stats
import scipy.interpolate
from skimage import measure


def plot_sigcluster_illustration_for_poster(sig_matrix, pac_matrix, channel_idx, condition_idx, n_phase, n_amplitude,
                                            max_cluster_size):
    """
    Plot the cluster and the corresponding PAC matrix for a given channel and condition idx. Use fonts for poster size.
    :param sig_matrix:
    :param pac_matrix:
    :param channel_idx:
    :param condition_idx:
    :param n_phase:
    :param n_amplitude:
    :param max_cluster_size:
    :return:
    """

    # plot the PAC matrix for poster
    fontsize = 20
    tick_steps = 3
    tick_size = 15
    plt.figure(figsize=(10, 8))
    plt.subplot(121)
    plt.imshow(sig_matrix[channel_idx, condition_idx,], origin='lower', interpolation='None')

    plt.xticks(np.linspace(0, n_phase, tick_steps), np.linspace(5, 35, tick_steps, dtype=int),
               fontsize=tick_size)
    plt.yticks(np.linspace(0, n_amplitude, tick_steps), np.linspace(150, 400, tick_steps, dtype=int),
               fontsize=tick_size)

    plt.xlabel('Phase frequency [Hz]', fontsize=fontsize)
    plt.ylabel('Amplitude frequency [Hz]', fontsize=fontsize)
    plt.title('Significance', fontsize=fontsize)

    plt.subplot(122)
    plt.imshow(pac_matrix[channel_idx, condition_idx,], origin='lower', interpolation='None',
               vmin=0.0, vmax=0.08)
    cbar = plt.colorbar(ticks=[0.02, 0.05, 0.08])
    cbar.ax.tick_params(labelsize=tick_size)
    plt.xlabel('Phase frequency [Hz]', fontsize=fontsize)
    plt.title('PAC', fontsize=fontsize)
    plt.xticks(np.linspace(0, n_phase, tick_steps), np.linspace(5, 35, tick_steps, dtype=int),
               fontsize=tick_size)
    plt.yticks(np.linspace(0, n_amplitude, tick_steps), np.linspace(150, 400, tick_steps, dtype=int),
               fontsize=tick_size)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_PATH_FIGURES, 'example_pac_cluster{}.pdf'.format(int(max_cluster_size))))
    plt.show()
    plt.close()


def plot_beta_band_selection_illustration_for_poster(pac_matrix_sig, pac_matrix_nonsig, sig_matrix1, sig_matrix2,
                                                     n_phase, n_amplitude,
                                                     f_phase, f_amp, mask, smoother_pac, max_idx, current_lfp_epochs,
                                                     subject_id, fs, save=False):

    # plot both, the sig and the smoothed pac mean
    fontsize = 20
    y_tick_steps = 3
    x_tick_steps = 5
    tick_size = 15
    legend_size = 15

    vmin = pac_matrix_sig.min()
    vmax = pac_matrix_sig.max()

    plt.figure(figsize=(15, 6))
    # plot the PAC matrix
    plt.subplot2grid((2, 4), (0, 0), rowspan=2)
    current_pac_data = pac_matrix_nonsig
    plt.imshow(current_pac_data, interpolation='None', origin='lower', vmin=vmin, vmax=vmax)

    # plot contours of significance
    contours = measure.find_contours(sig_matrix2, 0)
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=.5, color='orange')

    plt.xticks(np.linspace(0, n_phase, y_tick_steps),
               np.linspace(f_phase.min(), f_phase.max(), y_tick_steps, dtype=int),
               fontsize=tick_size)
    plt.yticks(np.linspace(0, n_amplitude, x_tick_steps),
               np.linspace(f_amp.min(), f_amp.max(), x_tick_steps, dtype=int),
               fontsize=tick_size)
    plt.xlabel('Phase frequency [Hz]', fontsize=fontsize)
    plt.ylabel('Amplitude frequency [Hz]', fontsize=fontsize)
    plt.title('PAC, discarded', fontsize=fontsize)
    plt.colorbar(pad=0.02, fraction=.09, ticks=np.round([vmin + 0.01, np.mean((vmin + 0.01, vmax - 0.01)), vmax - 0.01], 2))

    plt.subplot2grid((2, 4), (0, 1), rowspan=2)
    current_pac_data = pac_matrix_sig
    plt.imshow(current_pac_data, interpolation='None', origin='lower', vmin=vmin, vmax=vmax)
    # plot contours of significance
    contours = measure.find_contours(sig_matrix1, 0)
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=.5, color='orange')

    plt.xticks(np.linspace(0, n_phase, y_tick_steps),
               np.linspace(f_phase.min(), f_phase.max(), y_tick_steps, dtype=int),
               fontsize=tick_size)
    plt.yticks(np.linspace(0, n_amplitude, x_tick_steps),
               np.linspace(f_amp.min(), f_amp.max(), x_tick_steps, dtype=int),
               fontsize=tick_size)
    plt.xlabel('Phase frequency [Hz]', fontsize=fontsize)
    plt.title('PAC, accepted', fontsize=fontsize)
    plt.tight_layout()

    # plot the smoothed PAC values averaged over amplitude frequencies:
    plt.subplot2grid((2, 4), (0, 2), colspan=2)
    plt.title('Peak PAC and selected beta range', fontsize=fontsize)
    plt.plot(f_phase[mask], smoother_pac)
    plt.ylabel('PAC', fontsize=fontsize)
    xtick_vals_smoothed = np.linspace(f_phase[mask].min(), f_phase[mask].max(), x_tick_steps, dtype=int)
    plt.xticks(np.linspace(f_phase[mask].min(), f_phase[mask].max(), x_tick_steps),
               xtick_vals_smoothed,
               fontsize=tick_size)
    y_tick_steps = 3
    plt.yticks(np.linspace(smoother_pac.min(), smoother_pac.max(), y_tick_steps),
               np.round(np.linspace(smoother_pac.min(), smoother_pac.max(), y_tick_steps), 2),
               fontsize=tick_size)
    # plot the peak
    plt.axvline(f_phase[mask][max_idx], smoother_pac[max_idx], alpha=.3, color='C1')
    plt.plot(f_phase[mask][max_idx], smoother_pac[max_idx], 'o', markersize=8, label='peak PAC')
    plt.legend(frameon=False, prop={'size': legend_size})

    # plot the PSD of the corresponding LFP
    plt.subplot2grid((2, 4), (1, 2), colspan=2)
    # calculate psd for every epoch, average across epochs
    f_psd, psd = ut.calculate_psd(y=current_lfp_epochs[:, 0], fs=fs, window_length=1024)  # to get the dims
    for epoch_idx, lfp_epoch in enumerate(current_lfp_epochs[:, 1:].T):
        f_psd, psd_tmp = ut.calculate_psd(y=lfp_epoch, fs=fs, window_length=1024)
        psd += psd_tmp
    # divide by n epochs to average
    psd /= current_lfp_epochs.shape[1]
    # interpolate the psd to have the same sample point as in the PAC phase dimensions:
    psd_inter_f = scipy.interpolate.interp1d(f_psd, psd)
    psd = psd_inter_f(f_phase)

    plt.plot(f_phase[mask], psd[mask])

    plt.xticks(np.linspace(f_phase[mask].min(), f_phase[mask].max(), x_tick_steps),
               xtick_vals_smoothed,
               fontsize=tick_size)
    plt.yticks(np.linspace(psd[mask].min(), psd[mask].max(), y_tick_steps),
               np.round(np.linspace(psd[mask].min(), psd[mask].max(), y_tick_steps), 1),
               fontsize=tick_size)
    plt.xlabel('Frequency [Hz]', fontsize=fontsize)
    plt.ylabel('PSD', fontsize=fontsize)

    # plot the peak and the selected range
    plt.axvline(f_phase[mask][max_idx], smoother_pac[max_idx], alpha=.5, color='C1')
    fill_mask = ut.get_array_mask(f_phase >= (f_phase[mask][max_idx] - 6), f_phase <= (f_phase[mask][max_idx] + 6))
    plt.fill_between(f_phase, psd[mask].min(), psd[mask].max(), where=fill_mask, color='C1', alpha=0.1,
                     label='selected')
    plt.legend(prop={'size': legend_size}, frameon=False)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(SAVE_PATH_FIGURES, 'example_beta_range_selection_subj{}.pdf'.format(subject_id)))
    plt.show()


def plot_ratio_illustration_for_poster(fs, lfp_pre, lfp_band, zeros, extrema, extrema_kind, steepness_indices,
                                       steepness_values, save_figure=False):
    """
    Create a plot as in figure 3 of the Bernstein Conference poster to illustrate the filtering and feature extraction.
    The quantities needed here (zeros, extrema, extrema_kind, steepness_indices, steepness_values) you can calculate
    using methods in utils:
        - find_rising_and_falling_zeros
        - find_peaks_and_troughs_cole
        - calculate_rise_and_fall_steepness

    :param fs:
    :param lfp_pre: low pass filtered lfp signal
    :param lfp_band: beta band pass filtered lfp signal
    :param zeros: the zeros crossing indices of the array
    :param extrema: the extrama indices
    :param extrema_kind: max or min (1 or -1)
    :param steepness_indices: indices of maximal steepness
    :param steepness_values:
    :param save_figure:
    :return:
    """
    """
    Plot a figure for illustration of sharpness ratio and steepness ratio calculation. use large fonts for poster.
    """
    upto = 5
    fontsize = 20
    markersize = 8

    filter_offset = 167  # in ms
    samples_per_5ms = int(5 * fs / 1000)
    last_zero_idx = zeros[upto]
    extrema_to_use = extrema[:(upto - 1)]
    steepness_to_use = steepness_indices[:upto - 2]
    steepness_values_to_use = steepness_values[:upto - 2]
    sharpness_idx = np.sort(np.array([extrema_to_use - samples_per_5ms, extrema_to_use + samples_per_5ms]).flatten())

    extrema_idx = int((upto - 1) / 2)
    last_idx = np.max([last_zero_idx, extrema_to_use.max()])
    time_array = np.linspace(filter_offset, filter_offset + last_idx / fs * 1000, last_idx)
    plt.close()

    plt.figure(figsize=(15, 5))

    # plot LFP only up to last but one zero:
    plt.plot(time_array[:zeros[upto - 1]], lfp_pre[:zeros[upto - 1]], label='preprocessed')
    plt.plot(time_array[:zeros[upto - 1]], lfp_band[:zeros[upto - 1]], label='beta filtered', color='C1')
    plt.plot(time_array[zeros[:upto]], lfp_band[zeros[:upto]], 'o', color='C1', markersize=markersize)

    # plot zero line
    plt.axhline(0, color='grey')

    # plot +-5ms sharpness markers
    for idx, sharpness_sample in enumerate(sharpness_idx):
        # plt.axvline(x=time_array[sharpness_sample], color='k', alpha=.7, linewidth=.5)
        # color = 'red' if ll[idx] > 0 else 'blue'
        color = 'm'
        plt.plot(time_array[sharpness_sample], lfp_pre[sharpness_sample], '*', markersize=markersize, color=color)

    # plot maximum slope markers
    for idx, steepness_sample in enumerate(steepness_to_use):
        plt.plot(time_array[steepness_sample], lfp_pre[steepness_sample], 'd', markersize=markersize, color='g')
        # set up a tangent on this point
        m = (steepness_values_to_use[idx]) / 0.4
        y_val = lfp_pre[steepness_sample]
        x_val = time_array[steepness_sample]
        bias = y_val - m * x_val
        tangent = lambda x: m * x + bias
        tangent_time_array = np.linspace(time_array[steepness_sample] - 2,
                                         time_array[steepness_sample] + 2, 20)
        plt.plot(tangent_time_array, tangent(tangent_time_array), color='g')

    for idx, extrema in enumerate(extrema_to_use):
        if extrema_kind[idx] > 0:
            format = '^'
            label = 'peaks'
        else:
            format = 'v'
            label = 'troughs'

        color = 'red'
        plt.plot(time_array[extrema], lfp_pre[extrema], format, markersize=markersize, color=color)

    # remove labels for poster figure
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel('time', fontsize=fontsize)
    plt.ylabel('LFP', fontsize=fontsize)
    # plt.xlim([time_array[0], time_array[-1]])

    plt.tight_layout()
    plt.legend(frameon=False, prop={'size': 20})
    if save_figure:
        plt.savefig(os.path.join(SAVE_PATH_FIGURES, 'pre_sharpness.pdf'))
        plt.show()


def calculate_sig_channels_and_correlation_matrix(data_pairs_list, x_labels, y_labels, title_list, figure_filename_list,
                                                  n_bands, sig_subject_ids, outlier_std_factor, n_variables,
                                                  correlation_matrix_idx_l, correlation_matrix_idx_u, band_str, save_folder):
    """
    Calculates correlations for given sinusoidalness measures. Remove outliers, plot the results in scatter and line
    plots and return the corresponding correlations matrices, biases, r values, p values and standard errors.

    :param data_pairs_list: list with two dictionaries, holding the x and y values, respectively
    :param x_labels:
    :param y_labels:
    :param title_list:
    :param figure_filename_list:
    :param n_bands:
    :param sig_subject_ids:
    :param outlier_std_factor:
    :param n_variables:
    :param correlation_matrix_idx_l:
    :param correlation_matrix_idx_u:
    :param band_str:
    :param save_folder:
    :return:
    """

    # save the final correlation coefs in a matrix for better comparison
    n_corrs = int(n_variables * (n_variables - 1) / 2)
    correlation_matrix = np.ones(n_variables ** 2)
    slope_matrix = np.zeros(n_corrs)
    bias_matrix = np.zeros(n_corrs)
    p_matrix = np.zeros(n_corrs)

    data = []

    for data_pair_idx, data_pair in enumerate(data_pairs_list):
        plot_idx = 0
        d2, d1 = data_pair

        plt.figure(figsize=(10, 5))

        # extract the current data
        x_all = np.array(d1['all'])
        y_all = np.array(d2['all'])

        # regress all data points
        slope, bias, r, p, stderr = scipy.stats.linregress(x_all, y_all)
        r = round(r, 2)
        p = round(p, 3)

        # plot all data points, color coded for conditions
        plot_idx += 1
        plt.subplot(1, n_bands, plot_idx)
        plt.plot(np.array(d1['off']),
                 np.array(d2['off']), '*', markersize=5, label='off')
        plt.plot(np.array(d1['on']),
                 np.array(d2['on']), '*', markersize=5, label='on')
        plt.xlabel(x_labels[data_pair_idx])

        # only left plot gets ylabel
        if plot_idx == 1:
            plt.ylabel(y_labels[data_pair_idx])

        # plot two correlation lines, one for all points, one for selected points (outlier free)
        # plot all
        xvals = np.linspace(x_all.min(), x_all.max(), x_all.size)
        plt.plot(xvals, bias + slope * xvals, label='r={}, p={}'.format(r, p))

        # plot the regression line with outliers removed
        # use n times std away from mean as criterion
        x_clean, y_clean, x_out, y_out, mask = ut.exclude_outliers(x_all, y_all,
                                                                   n=outlier_std_factor)

        # save the cleaned data
        data.append([x_clean, y_clean])

        # plot the outliers
        outlier_indices = np.where(np.logical_not(mask))[0]
        outlier_labels = sig_subject_ids
        for outlier_idx in range(outlier_indices.shape[0]):
            plt.plot(x_out[outlier_idx], y_out[outlier_idx], '+', markersize=7,
                     label=outlier_labels[outlier_indices[outlier_idx]])

        # plot the new regression line
        slope, bias, r, p, stderr = scipy.stats.linregress(x_clean, y_clean)
        xvals = np.linspace(x_clean.min(), x_clean.max(), x_clean.size)
        plt.plot(xvals, bias + slope * xvals, label='r={}, p={}, cleaned'.format(round(r, 2), round(p, 4)))

        correlation_matrix[correlation_matrix_idx_l[data_pair_idx]] = r
        correlation_matrix[correlation_matrix_idx_u[data_pair_idx]] = r
        bias_matrix[data_pair_idx] = bias
        slope_matrix[data_pair_idx] = slope
        p_matrix[data_pair_idx] = p

        plt.legend(loc=1, prop={'size': 7})
        # plt.ylim(ylim)
        plt.title('{} n={}'.format(band_str[0], x_all.size))

        plt.suptitle(title_list[data_pair_idx])
        figure_filename = figure_filename_list[data_pair_idx]
        plt.savefig(os.path.join(save_folder, figure_filename))
        # plt.show()
        plt.close()

    correlation_matrix = np.reshape(correlation_matrix, (n_variables, n_variables))

    return correlation_matrix, bias_matrix, slope_matrix, p_matrix, data


def plot_correlation_matrix(corr, variable_labels, save_folder):

    vmin = -1.
    vmax = 1.
    tick_size = 15
    fontsize = 20
    n_variables = corr.shape[0]

    print(corr)
    # mask the matrix where needed
    corr[np.tril_indices(n_variables)] = 1
    corrm = np.ma.masked_equal(corr, 1)

    plt.imshow(corrm, interpolation=None, origin='upper', cmap='viridis', vmax=vmax, vmin=vmin, alpha=.8)
    plt.xticks(np.arange(n_variables), variable_labels, fontsize=fontsize)
    plt.gca().yaxis.tick_right()
    plt.yticks(np.arange(n_variables), variable_labels, fontsize=fontsize)
    plt.gca().xaxis.tick_top()

    cbar = plt.colorbar(ticks=[-vmin, 0, vmax], pad=.15)
    cbar.ax.tick_params(labelsize=tick_size)

    plt.tight_layout()

    plt.savefig(os.path.join(save_folder, 'correlation_matrix_{}.pdf'.format(n_variables)))
    # plt.show()
    plt.close()
