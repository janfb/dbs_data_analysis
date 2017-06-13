import numpy as np
import matplotlib.pyplot as plt
import os
from definitions import SAVE_PATH_FIGURES
import utils as ut
import scipy.stats


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
    plt.figure(figsize=(11, 8))
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


def plot_ratio_illustration_for_poster(fs, lfp_pre, lfp_band, zeros, extrema, extrema_kind, steepness_indices):
    """
    Plot a figure for illustration of sharpness ratio and steepness ration calculation. use large fonts for poster.
    :param fs:
    :param lfp_pre:
    :param lfp_band:
    :param zeros:
    :param extrema:
    :param extrema_kind:
    :param steepness_indices:
    :return:
    """
    upto = 7
    fontsize = 20
    markersize = 8

    filter_offset = 167  # in ms
    samples_per_5ms = int(5 * fs / 1000)
    last_zero_idx = zeros[upto]
    extrema_to_use = extrema[:(upto - 1)]
    steepness_to_use = steepness_indices[:upto - 2]
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
    ll = extrema_kind.repeat(2)

    # plot +-5ms sharpness markers
    for idx, sharpness_sample in enumerate(sharpness_idx):
        # plt.axvline(x=time_array[sharpness_sample], color='k', alpha=.7, linewidth=.5)
        # color = 'red' if ll[idx] > 0 else 'blue'
        color = 'm'
        plt.plot(time_array[sharpness_sample], lfp_pre[sharpness_sample], '*', markersize=markersize, color=color)

    # plot maximum slope markers
    for steepness_sample in steepness_to_use:
        plt.plot(time_array[steepness_sample], lfp_pre[steepness_sample], 'd', markersize=markersize, color='g')

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
    plt.legend(prop={'size': 20})
    plt.savefig(os.path.join(SAVE_PATH_FIGURES, 'pre_sharpness.pdf'))
    plt.show()


def calculate_sig_channels_and_correlation_matrix(data_pairs_list, x_labels, y_labels, title_list, figure_filename_list,
                                                  n_bands, sig_subject_ids, outlier_std_factor, n_variables,
                                                  correlation_matrix_idx_l, correlation_matrix_idx_u, band_str, save_folder):

    # save the final correlation coefs in a matrix for better comparison
    n_corrs = n_variables * (n_variables - 1) / 2
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

        # plot two correlation line, one for all points, one for selected points (outlier free)
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

    vmin = -0.5
    vmax = 0.5
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
