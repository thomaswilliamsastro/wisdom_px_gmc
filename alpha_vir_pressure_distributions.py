# -*- coding: utf-8 -*-
"""
KDE plot for the virial parameter nad pressure

@author: Tom Williams
"""

import itertools

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
from scipy.stats import gaussian_kde
from astropy.io import fits
from matplotlib.pyplot import cm

from vars import wisdom_dir, plot_dir, galaxy_dict, mask, vel_res, co_conv_factors


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14
sns.set_color_codes()

os.chdir(wisdom_dir)

correct_for_inclination = False

channel_corr_dir = 'channel_corr'

vel_res_int = float(vel_res.strip('kms').replace('p', '.'))

G = 4.301e-3
f = 10 / 9

target_resolutions = ['60pc', '90pc', '120pc']

parameters = ['alpha_vir']

for parameter in parameters:

    if not os.path.exists(os.path.join(plot_dir, parameter)):
        os.makedirs(os.path.join(plot_dir, parameter))

    parameter_dict = {}

    if parameter == 'alpha_vir':
        xlabel = r'$\alpha_\mathrm{vir}$'
        xlim = [10**-2.1, 10**2.1]
    elif parameter == 'pressure':
        xlabel = r'$P_\mathregular{turb}$ (K cm$^{-3}$)'
        xlim = [10**2, 10**9]
    else:
        raise Warning('I dunno what a %s is' % parameter)

    for galaxy in galaxy_dict.keys():

        plot_name = os.path.join(plot_dir, parameter, '%s_%s' % (galaxy, parameter))

        colour = itertools.cycle(sns.color_palette('deep'))

        co_line = galaxy_dict[galaxy]['co_line']
        antenna_config = galaxy_dict[galaxy]['antenna_config']

        if correct_for_inclination:
            inc = np.radians(galaxy_dict[galaxy]['info']['inc'])
            inc_factor = np.cos(inc)
        else:
            inc_factor = 1

        plt.figure(figsize=(5, 4))

        for target_resolution in target_resolutions:

            c = next(colour)

            # Read in mom0/ew

            vel_disp_file_name = os.path.join('new_reduction', 'derived', galaxy,
                                              '%s_%s_%s_%s_%s_%s_ew.fits' % (
                                                  galaxy, antenna_config, co_line, vel_res, target_resolution, mask))
            if not os.path.exists(vel_disp_file_name):
                continue
            surf_dens_file_name = vel_disp_file_name.replace('_ew', '_mom0')

            vel_disp_hdu = fits.open(vel_disp_file_name)[0]
            surf_dens_hdu = fits.open(surf_dens_file_name)[0]

            beam_fwhm = surf_dens_hdu.header['BMAJ']
            pix_size = np.abs(surf_dens_hdu.header['CDELT1'])

            pix_per_beam = beam_fwhm / pix_size
            areal_oversampling = np.pi / (4 * np.log(2)) * pix_per_beam ** 2

            co_conv_factor = co_conv_factors[co_line]

            vel_disp = vel_disp_hdu.data
            surf_dens = surf_dens_hdu.data * co_conv_factor

            # Read in and correct velocity dispersion

            channel_corr_file_name = os.path.join(channel_corr_dir, '%s_%s_%s_%s_%s_corr.npy' %
                                                  (galaxy, antenna_config, co_line, vel_res, target_resolution))
            r = np.load(channel_corr_file_name)

            k = 0.47 * r - 0.23 * r ** 2 - 0.16 * r ** 3 + 0.43 * r ** 4
            sigma_resp = vel_res_int / np.sqrt(2 * np.pi) * (1 + 1.8 * k + 10.4 * k ** 2)
            vel_disp = np.sqrt(vel_disp ** 2 - sigma_resp ** 2)

            r_beam = float(target_resolution.strip('pc')) / 2

            if parameter == 'alpha_vir':
                param_val = 5.77 * vel_disp ** 2 * (surf_dens * inc_factor) ** -1 * (r_beam / 40) ** -1
            elif parameter == 'pressure':
                param_val = 61.3 * (surf_dens * inc_factor) * vel_disp ** 2 * (r_beam / 40) ** -1
            else:
                raise Warning('I dunno what a %s is' % parameter)

            idx = np.where(~np.isnan(param_val))
            param_val = param_val[idx]
            surf_dens = surf_dens[idx]

            # Calculate KDE distribution

            kde_range = np.arange(np.log10(xlim[0]), np.log10(xlim[1]), 0.01)
            kde = gaussian_kde(np.log10(param_val), weights=surf_dens, bw_method='silverman')
            kde_hist = kde.evaluate(kde_range)

            percentiles = weighted_quantile(param_val, quantiles=[0.16, 0.5, 0.84], sample_weight=surf_dens)

            parameter_dict[galaxy + target_resolution] = percentiles

            plt.plot(10 ** kde_range, kde_hist, label=target_resolution, c=c)
            plt.axvline(percentiles[1], ls='-', c=c)
            for percentile in [percentiles[0], percentiles[-1]]:
                plt.axvline(percentile, ls='--', c=c)
            plt.xscale('log')

        plt.legend(loc='upper left', frameon=False)

        plt.xlim(xlim)
        ylim = plt.ylim()
        plt.ylim(0, ylim[-1])

        plt.xlabel(xlabel)
        plt.ylabel('Probability Density')

        plt.tight_layout()

        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.savefig(plot_name + '.pdf', bbox_inches='tight')

        plt.close()
        # plt.show()

    plot_name = os.path.join(plot_dir, parameter + '_distribution')

    plt.figure(figsize=(5, 8))
    ax1 = plt.subplot(1, 1, 1)

    frame1 = plt.gca()

    position = 0.1

    marker = ['o', 's', '^', 'p']
    # plot_colour = iter(cm.rainbow(np.linspace(0, 1, len(galaxies))))
    colour = itertools.cycle(sns.color_palette('deep'))

    y_position = 1 / (len(target_resolutions) + 1)
    positions = [y_position * (i+1) for i in range(len(target_resolutions))]

    y_tick_positions = [(2 * i + 1)/2 for i in range(len(galaxy_dict.keys()))]
    y_tick_labels = [galaxy.upper() for galaxy in galaxy_dict.keys()]

    for i, galaxy in enumerate(galaxy_dict.keys()):

        c = next(colour)

        for j, target_resolution in enumerate(target_resolutions):

            if i == 0:
                plt.scatter(-100, -100, c='k', marker=marker[j], label=target_resolution)

            # Divide off each galaxy

            if i != 0:
                plt.axhline(i, c='gray', ls='--')
            if i != len(galaxy_dict.keys()) - 1:
                plt.axhline(i + 1, c='gray', ls='--')

            if not galaxy + target_resolution in parameter_dict:
                continue

            xerr = np.array([[parameter_dict[galaxy + target_resolution][0],
                             parameter_dict[galaxy + target_resolution][-1]]]).T

            plt.errorbar(parameter_dict[galaxy + target_resolution][1], i + positions[-j - 1],
                         xerr=xerr,
                         color=c, ls='none', marker=marker[j])

    if parameter == 'alpha_vir':
        plt.axvline(1, c='k', ls='--')

    plt.xscale('log')

    plt.xlim(xlim)
    plt.ylim(0, len(y_tick_labels))

    plt.yticks(ticks=y_tick_positions, labels=y_tick_labels)

    plt.xlabel(xlabel)

    plt.legend(loc='upper left', frameon=False)

    plt.tight_layout()

    plt.savefig(plot_name + '.png', bbox_inches='tight')
    plt.savefig(plot_name + '.pdf', bbox_inches='tight')

    plt.close()
    # plt.show()

print('Complete!')
