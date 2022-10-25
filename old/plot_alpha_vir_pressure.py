# -*- coding: utf-8 -*-
"""
KDE plot for the virial parameter nad pressure

@author: Tom Williams
"""

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
from astropy.io import fits
from matplotlib.pyplot import cm

from vars import wisdom_dir, plot_dir, galaxies, resolutions

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(wisdom_dir)

with open('resolutions.pkl', 'rb') as resolution_file:
    resolution_dict = pickle.load(resolution_file)

parameters = ['alpha_vir', 'pressure']
parameters = ['alpha_vir']

for parameter in parameters:

    if not os.path.exists(os.path.join(plot_dir, parameter)):
        os.makedirs(os.path.join(plot_dir, parameter))

    parameter_dict = {}

    # galaxies = galaxies[:4]

    for galaxy in galaxies:

        parameter_dict[galaxy] = []
        parameter_dict[galaxy + '_err_up'] = []
        parameter_dict[galaxy + '_err_down'] = []

        plt.figure(figsize=(14, 4))

        # plt.suptitle(galaxy)

        gs = gridspec.GridSpec(1, len(resolutions))
        gs.update(wspace=0, hspace=0)

        subplot_idx = 0

        native_resolution = resolution_dict[galaxy]

        for resolution in resolutions:

            # Read in the relevant FITs files

            try:
                vel_disp_hdu = fits.open('regrids/' + galaxy + '_' + str(resolution) + '_eff_width.fits')[0]
            except FileNotFoundError:
                parameter_dict[galaxy].append(np.nan)
                parameter_dict[galaxy + '_err_up'].append(np.nan)
                parameter_dict[galaxy + '_err_down'].append(np.nan)
                continue
            sigma_gas_hdu = fits.open('regrids/' + galaxy + '_' + str(resolution) + '_surf_dens.fits')[0]

            # Flatten and get rid of NaNs

            vel_disp = vel_disp_hdu.data.flatten()
            sigma_gas = sigma_gas_hdu.data.flatten()

            idx = np.where((np.isnan(vel_disp) == False) & (np.isinf(vel_disp) == False) & \
                           (np.isnan(sigma_gas) == False) & (np.isinf(sigma_gas) == False))

            vel_disp = vel_disp[idx]
            sigma_gas = sigma_gas[idx]

            idx = np.where(vel_disp > 0)

            vel_disp = vel_disp[idx]
            sigma_gas = sigma_gas[idx]

            # Correct the velocity dispersions given the correlation

            r_corr = np.loadtxt('regrids/' + galaxy + '_' + str(resolution) + '_correlation.txt', usecols=0)

            v_channel = 2
            if galaxy == 'NGC4429':
                v_channel = 3

            k = 0.47 * r_corr - 0.23 * r_corr ** 2 - 0.16 * r_corr ** 3 + 0.43 * r_corr ** 4
            sigma_response = (v_channel / np.sqrt(2 * np.pi)) * (1 + 1.18 * k + 10.4 * k ** 2)
            vel_disp = np.sqrt(vel_disp ** 2 - sigma_response ** 2)

            idx = np.where(~np.isnan(vel_disp))
            sigma_gas = sigma_gas[idx]
            vel_disp = vel_disp[idx]

            if len(vel_disp) == 0:
                continue

            # Calculate the virial parameter

            try:
                r_beam = resolution / 2
            except TypeError:
                r_beam = native_resolution / 2

            if parameter == 'alpha_vir':

                param_val = 5.77 * vel_disp ** 2 * sigma_gas ** -1 * (r_beam / 40) ** -1

            elif parameter == 'pressure':

                param_val = 61.3 * sigma_gas * vel_disp ** 2 * (r_beam / 40) ** -1

            param_val = param_val[param_val > 0]

            parameter_dict[galaxy].append(np.nanmedian(param_val))
            parameter_dict[galaxy + '_err_up'].append(np.nanpercentile(param_val, 84) - np.nanmedian(param_val))
            parameter_dict[galaxy + '_err_down'].append(np.nanmedian(param_val) - np.nanpercentile(param_val, 16))

            ax = plt.subplot(gs[subplot_idx])

            x_log, y = sns.kdeplot(np.log10(param_val),
                                   bw=0.1).get_lines()[0].get_data()

            ax.clear()

            plt.plot(10 ** x_log, y,
                     c='k',
                     lw=2)
            plt.fill_between(10 ** x_log, y,
                             color='k',
                             alpha=0.3)

            plt.axvline(np.nanmedian(param_val),
                        c='k',
                        lw=2,
                        ls='-')

            if parameter == 'alpha_vir':
                plt.axvline(1, c='k', lw=2, ls='--')

            ax.axvspan(np.nanpercentile(param_val, 16), np.nanpercentile(param_val, 84),
                       alpha=0.5, color='k')

            plt.xscale('log')

            if parameter == 'alpha_vir':
                xlabel = r'$\alpha_\mathregular{vir,' + str(resolution) + '}$'
            elif parameter == 'pressure':
                xlabel = r'$P_\mathregular{turb, ' + str(resolution) + '}$ (K cm$^{-3}$)'

            plt.xlabel(xlabel)

            if subplot_idx == 0:
                plt.ylabel('Probability density')

            plt.yticks([])

            if parameter == 'alpha_vir':
                xlim = (10 ** -2.1, 10 ** 2.1)
            else:
                xlim = (10 ** 1.1, 10 ** 9.9)

            plt.ylim(bottom=0)
            plt.xlim(xlim)

            subplot_idx += 1

        plt.tight_layout()

        plot_name = os.path.join(plot_dir, parameter, parameter + '_' + galaxy)

        plt.savefig(plot_name + '.png',
                    bbox_inches='tight')
        plt.savefig(plot_name + '.pdf',
                    bbox_inches='tight')

        plt.close()

    plt.close('all')

    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(1, 1, 1)

    frame1 = plt.gca()

    position = 0.1

    marker = ['o', 's', '^', 'p']
    plot_colour = iter(cm.rainbow(np.linspace(0, 1, len(galaxies))))

    for galaxy in galaxies:

        c = next(plot_colour)

        for i in range(len(parameter_dict[galaxy])):

            err_down = parameter_dict[galaxy + '_err_down'][i]
            err_up = parameter_dict[galaxy + '_err_up'][i]

            errors = np.array([[err_down, err_up]]).T

            if galaxy == galaxies[0]:

                if i == 0:
                    plt.scatter(0, 0, c='k', marker=marker[i],
                                label='Native\nresolution')

                else:
                    plt.scatter(0, 0, c='k', marker=marker[i],
                                label=str(resolutions[i]) + 'pc')

            plt.errorbar(parameter_dict[galaxy][i],
                         position + i / 10,
                         xerr=errors,
                         c=c,
                         marker=marker[i])

            # Divide off each galaxy

        plt.axhline(position + 0.4,
                    c='k',
                    lw=1)

        if parameter == 'alpha_vir':
            text_pos = 1e-3
        else:
            text_pos = 10

        plt.text(text_pos, position + 0.15, galaxy,
                 ha='center',
                 va='center',
                 fontsize=14)

        position += 0.5

    plt.axvline(1, c='k', ls='--')

    if parameter == 'alpha_vir':
        xlabel = r'$\alpha_\mathregular{vir}$'
    elif parameter == 'pressure':
        xlabel = r'$P_\mathregular{turb}$ (K cm$^{-3}$)'

    plt.xlabel(xlabel)

    if parameter == 'alpha_vir':
        xlim = (0.01, 100)
    else:
        xlim = (10 ** 2.1, 10 ** 8.5)

    plt.ylim([0, position - 0.1])
    plt.xlim(xlim)

    plt.xscale('log')

    frame1.axes.get_yaxis().set_visible(False)

    ax1.legend(numpoints=1,
               frameon=False,
               bbox_to_anchor=(1.1, 0.5),
               loc='center left')

    plt.tight_layout()

    # plt.show()

    plot_name = os.path.join(plot_dir, parameter, parameter + '_summary')

    plt.savefig(plot_name + '.png',
                bbox_inches='tight')
    plt.savefig(plot_name + '.pdf',
                bbox_inches='tight')

    plt.close()

print('Complete!')
