import os

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from astropy.io import fits
import seaborn as sns
from spectral_cube import SpectralCube

from vars import wisdom_dir, plot_dir, galaxy_dict, mask, vel_res, co_conv_factors

from external.sun_cube_tools import calc_channel_corr, censoring_function


def collapse_dict(dict_to_collapse):
    dict_collapsed = np.array([])
    for key in dict_to_collapse.keys():
        dict_collapsed = np.append(dict_collapsed, dict_to_collapse[key])

    return dict_collapsed


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14
sns.set_color_codes()

os.chdir(wisdom_dir)

sigma_plot_dir = os.path.join(plot_dir, 'sigma_Sigma')
if not os.path.exists(sigma_plot_dir):
    os.makedirs(sigma_plot_dir)

channel_corr_dir = os.path.join('channel_corr')
if not os.path.exists(channel_corr_dir):
    os.makedirs(channel_corr_dir)
overwrite_channel_corr = False

vel_res_int = float(vel_res.strip('kms').replace('p', '.'))

alpha_vir = 1
G = 4.301e-3
f = 10 / 9

vel_disp_dict = {}
surf_dens_dict = {}

target_resolutions = ['60pc', '90pc', '120pc']
surf_dens_lims = [0.8, 12000]
vel_disp_lims = [0.5, 110]

for target_resolution in target_resolutions:
    vel_disp_dict[target_resolution] = {}
    surf_dens_dict[target_resolution] = {}

for galaxy in galaxy_dict.keys():

    co_line = galaxy_dict[galaxy]['co_line']
    antenna_config = galaxy_dict[galaxy]['antenna_config']

    plot_name = os.path.join(sigma_plot_dir, galaxy + '_sigma_Sigma')

    fig, axes = plt.subplots(figsize=(8, 4), nrows=1, ncols=len(target_resolutions))

    for i, target_resolution in enumerate(target_resolutions):

        target_resolution_kpc = float(target_resolution.strip('pc'))

        if target_resolution not in galaxy_dict[galaxy]['resolutions']:
            continue

        vel_disp_file_name = os.path.join('new_reduction', 'derived', galaxy,
                                          '%s_%s_%s_%s_%s_%s_ew.fits' % (
                                              galaxy, antenna_config, co_line, vel_res, target_resolution, mask))
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

        # Correct velocity distribution for line broadening. Since this is quite demanding, save out after calculating.

        channel_corr_file_name = os.path.join(channel_corr_dir, '%s_%s_%s_%s_%s_corr.npy' %
                                              (galaxy, antenna_config, co_line, vel_res, target_resolution))

        if not os.path.exists(channel_corr_file_name) or overwrite_channel_corr:

            print('Calculating channel-to-channel correlation')

            mask_file_name = os.path.join('new_reduction', 'derived', galaxy,
                                          '%s_%s_%s_%s_%s_strictmask.fits' %
                                          (galaxy, antenna_config, co_line, vel_res, target_resolution))
            cube_mask = fits.open(mask_file_name)[0].data.astype(bool)

            cube_file_name = os.path.join('new_reduction', 'derived', galaxy,
                                          '%s_%s_%s_%s_%s.fits' %
                                          (galaxy, antenna_config, co_line, vel_res, target_resolution))
            cube = SpectralCube.read(cube_file_name)

            channel_corr_mask = ~cube_mask
            channel_corr_mask[np.isnan(cube.hdu.data)] = False

            r, _ = calc_channel_corr(cube, mask=channel_corr_mask)
            np.save(channel_corr_file_name, r)

        else:

            r = np.load(channel_corr_file_name)

        k = 0.47 * r - 0.23 * r ** 2 - 0.16 * r ** 3 + 0.43 * r ** 4
        sigma_resp = vel_res_int / np.sqrt(2 * np.pi) * (1 + 1.8 * k + 10.4 * k ** 2)
        vel_disp = np.sqrt(vel_disp ** 2 - sigma_resp ** 2)

        # Also calculate the completeness limit from the noise cube

        noise_cube_file_name = os.path.join('new_reduction', 'derived', galaxy,
                                            '%s_%s_%s_%s_%s_noise.fits' %
                                            (galaxy, antenna_config, co_line, vel_res, target_resolution))
        noise_cube = fits.open(noise_cube_file_name)[0].data
        noise = np.nanmedian(noise_cube)

        test_vel_disp = np.linspace(vel_disp_lims[0], vel_disp_lims[1])

        # Kernel rounding errors
        k = k.astype(np.float32)

        test = censoring_function(test_vel_disp, vel_res_int, noise, spec_resp_kernel=[k, 1-2*k, k], snr_crit=3)
        test *= co_conv_factor

        # TODO: Fit points with power law

        # Calculate virialised lines
        surf_dens_vir = np.linspace(surf_dens_lims[0], surf_dens_lims[1], 1000)
        vel_disp_vir = (f * alpha_vir * G / 5 * np.pi / np.log(2)) ** 0.5 * \
                       (target_resolution_kpc / 2) ** 0.5 * surf_dens_vir ** 0.5

        axes[i].scatter(surf_dens, vel_disp, c='b', s=2)
        axes[i].plot(surf_dens_vir, vel_disp_vir, c='k', ls='--')
        axes[i].axhline(vel_res_int, c='k', ls=':')
        # axes[i].plot(test_mol_sens_limit, test_vel_disp_limit, c='k', ls='--')
        axes[i].plot(test, test_vel_disp, c='r', ls='-.')
        axes[i].set_xscale('log')
        axes[i].set_yscale('log')

        axes[i].set_ylim(vel_disp_lims)
        axes[i].set_xlim(surf_dens_lims)

        if i == 0:
            axes[i].set_ylabel(r'$\sigma_\mathrm{%s}\,(\mathrm{km\,s^{-1}})$' % target_resolution)
        if i > 0:
            axes[i].tick_params(labelleft=False)
        if i == len(target_resolutions) - 1:
            axes[i].set_ylabel(r'$\sigma_\mathrm{%s}\,(\mathrm{km\,s^{-1}})$' % target_resolution)
            axes[i].tick_params(right=True, labelright=True, which='both')
            axes[i].yaxis.set_label_position('right')

        axes[i].set_xlabel(r'$\Sigma_\mathrm{%s}\,(\mathrm{M}_\odot\,\mathrm{pc}^{-2})$' % target_resolution)

    plt.suptitle(galaxy.upper())

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)

    # plt.show()
    plt.savefig(plot_name + '.pdf', bbox_inches='tight')
    plt.savefig(plot_name + '.png', bbox_inches='tight')
    plt.close()

print('Complete!')
