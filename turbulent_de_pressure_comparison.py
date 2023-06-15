# -*- coding: utf-8 -*-
"""
Compare internal turbulent to dynamical equilibrium pressure

@author: Tom Williams
"""
import copy
import os
import warnings

import cmocean
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astroquery.ipac.ned import Ned
from mpl_toolkits.axes_grid1 import make_axes_locatable
from spectral_cube import SpectralCube

from external.sun_cube_tools import calc_channel_corr
from vars import wisdom_dir, plot_dir, galaxy_dict, vel_res, mask, co_conv_factors, zoom

warnings.simplefilter('ignore')

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(wisdom_dir)

pressure_plot_dir = os.path.join(plot_dir, 'pressure_comparison')
if not os.path.exists(pressure_plot_dir):
    os.makedirs(pressure_plot_dir)

G = 6.67e-11
kb = 1.38e-23

channel_corr_dir = 'channel_corr'

vel_res_int = float(vel_res.strip('kms').replace('p', '.'))

pressure_ratio_radial_bins = {}
pressure_ratio_radial_values = {}

for galaxy in galaxy_dict.keys():

    print(galaxy)

    co_line = galaxy_dict[galaxy]['co_line']
    co_conv_factor = co_conv_factors[co_line]
    antenna_config = galaxy_dict[galaxy]['antenna_config']

    dist = galaxy_dict[galaxy]['info']['dist']
    pc_conversion_fac = 4.84 * dist  # pc/arcsec

    try:
        galaxy_info = galaxy_dict[galaxy]['info']
    except KeyError:
        continue

    pa = galaxy_info['pa']
    inc = galaxy_info['inc']
    ml = np.array(galaxy_info['ml'])
    surf = np.array(galaxy_info['surf'])
    sigma_arcsec = np.array(galaxy_info['sigma_arcsec'])
    qobs = np.array(galaxy_info['qobs'])
    r_eff = galaxy_info['reff'] * pc_conversion_fac

    # Read in mom0/ew

    vel_disp_file_name = os.path.join('new_reduction', 'derived', galaxy,
                                      '%s_%s_%s_%s_%s_ew.fits' % (
                                          galaxy, antenna_config, co_line, vel_res, mask))
    if not os.path.exists(vel_disp_file_name):
        continue
    surf_dens_file_name = vel_disp_file_name.replace('_ew', '_mom0')

    vel_disp_hdu = fits.open(vel_disp_file_name)[0]
    surf_dens_hdu = fits.open(surf_dens_file_name)[0]

    wcs_orig = WCS(vel_disp_hdu)

    vel_disp = vel_disp_hdu.data
    surf_dens = surf_dens_hdu.data * co_conv_factor

    beam_fwhm = surf_dens_hdu.header['BMAJ']
    pix_size = np.abs(surf_dens_hdu.header['CDELT1'])
    resolution = pc_conversion_fac * beam_fwhm * 3600

    # Read in and correct velocity dispersion

    channel_corr_file_name = os.path.join(channel_corr_dir, '%s_%s_%s_%s_corr.npy' %
                                          (galaxy, antenna_config, co_line, vel_res))

    if not os.path.exists(channel_corr_file_name):

        print('Calculating channel-to-channel correlation')

        mask_file_name = os.path.join('new_reduction', 'derived', galaxy,
                                      '%s_%s_%s_%s_strictmask.fits' %
                                      (galaxy, antenna_config, co_line, vel_res))
        cube_mask = fits.open(mask_file_name)[0].data.astype(bool)

        cube_file_name = os.path.join('new_reduction', 'derived', galaxy,
                                      '%s_%s_%s_%s.fits' %
                                      (galaxy, antenna_config, co_line, vel_res))
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

    r_beam = resolution / 2

    p_turb = 61.3 * surf_dens * vel_disp ** 2 * (r_beam / 40) ** -1

    result_table = Ned.query_object(galaxy.upper())
    ra, dec = result_table['RA'][0], result_table['DEC'][0]
    w = WCS(vel_disp_hdu)
    x_cen, y_cen = w.all_world2pix(ra, dec, 1)

    # Turn the MGE into a stellar mass surface density for each pixel

    yi, xi = np.meshgrid((np.arange(surf_dens.shape[1]) - x_cen) * pix_size * pc_conversion_fac * 3600,
                         (np.arange(surf_dens.shape[0]) - y_cen) * pix_size * pc_conversion_fac * 3600)

    # Project into galaxy plane

    angle = np.radians(pa)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    x_proj = xi * cos_a + yi * sin_a
    y_proj = - xi * sin_a + yi * cos_a
    x_proj /= np.cos(np.radians(inc))

    r = np.sqrt(x_proj ** 2 + y_proj ** 2)

    sigma = sigma_arcsec * pc_conversion_fac
    qintr2 = qobs ** 2  # - np.cos(np.radians(inc)) ** 2
    qintr = np.sqrt(qintr2) / np.sin(np.radians(inc))
    surfcube = (surf * qobs) / (qintr * sigma * np.sqrt(2 * np.pi))

    stellar_dens = np.zeros_like(surf_dens)

    for i in range(len(sigma)):
        stellar_dens += ((surfcube[i] * ml) * np.exp(((-1.) / (2 * (sigma[i] ** 2))) * (r ** 2)))

    # Convert things through to SI units

    surf_dens_si = surf_dens * 2e30 / 3.086e16 ** 2
    stellar_dens_si = stellar_dens * 2e30 / 3.086e16 ** 3

    # Assume we're molecular dominated, and use the Sun+ (2020) prescription

    pressure_equilibrium = (np.pi * G / 2) * surf_dens_si ** 2 + \
                           surf_dens_si * np.sqrt(2 * G * stellar_dens_si) * vel_disp
    pressure_equilibrium /= kb * 1e6

    pressure_obs = 61.3 * surf_dens * vel_disp ** 2 * (r_beam / 40) ** -1

    pressure_ratio = np.log10(pressure_obs / pressure_equilibrium)

    # Create radial profiles of this pressure ratio

    max_dist = 0.4 * r_eff

    radial_bins = np.linspace(0, max_dist, 10)
    bin_width = (radial_bins[1] - radial_bins[0]) / 2

    pressure_ratio_radial_bins[galaxy] = radial_bins / r_eff

    radial_values = np.zeros([len(radial_bins), 3])

    for i, radial_bin in enumerate(radial_bins):
        idx = np.where((radial_bin - bin_width <= r) & (r <= radial_bin + bin_width))
        radial_values[i, :] = np.nanpercentile(pressure_ratio[idx], [16, 50, 84])
    radial_values[:, 0] = radial_values[:, 1] - radial_values[:, 0]
    radial_values[:, 2] -= radial_values[:, 1]

    pressure_ratio_radial_values[galaxy] = radial_values

    # Plot maps for the pressure ratio

    plot_name = os.path.join(pressure_plot_dir, galaxy + '_pressure_comp_map')

    plt.figure(figsize=(12, 4))

    # First, turbulent pressure

    if galaxy in zoom.keys():

        position = zoom[galaxy]['centre']
        if position is None:
            position = np.asarray(pressure_obs.shape) / 2
        size = zoom[galaxy]['zoom']

        pressure_obs = Cutout2D(pressure_obs,
                                wcs=wcs_orig,
                                position=position,
                                size=size)
        wcs = pressure_obs.wcs
        pressure_obs = pressure_obs.data
    else:
        wcs = copy.deepcopy(wcs_orig)

    vmin, vmax = np.nanpercentile(np.log10(pressure_obs), [2.5, 97.5])

    ax = plt.subplot(1, 3, 1, projection=wcs)

    im = plt.imshow(np.log10(pressure_obs),
                    origin='lower',
                    interpolation='none',
                    vmin=vmin, vmax=vmax,
                    cmap='viridis')

    plt.grid()

    plt.text(0.05, 0.95, galaxy.upper(),
             ha='left', va='top',
             bbox=dict(facecolor='white', edgecolor='k'),
             transform=ax.transAxes)

    ax.coords[0].set_axislabel('RA (J2000)')
    ax.coords[1].set_axislabel('Dec (J2000)')

    # ax.coords[0].set_ticklabel(rotation=45, pad=40)
    ax.coords[0].set_major_formatter('hh:mm:ss')

    ax.coords[0].display_minor_ticks(True)
    ax.coords[1].display_minor_ticks(True)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0, axes_class=matplotlib.axes.Axes)
    plt.colorbar(im, cax=cax, label=r'$\log_{10}(P_\mathrm{turb}~[\mathrm{K~cm^{-3}}])$',
                 orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    # Second, dynamical equilibrium pressure

    if galaxy in zoom.keys():

        position = zoom[galaxy]['centre']
        if position is None:
            position = np.asarray(pressure_equilibrium.shape) / 2
        size = zoom[galaxy]['zoom']

        pressure_equilibrium = Cutout2D(pressure_equilibrium,
                                        wcs=wcs_orig,
                                        position=position,
                                        size=size)
        wcs = pressure_equilibrium.wcs
        pressure_equilibrium = pressure_equilibrium.data
    else:
        wcs = copy.deepcopy(wcs_orig)

    vmin, vmax = np.nanpercentile(np.log10(pressure_equilibrium), [2.5, 97.5])

    ax = plt.subplot(1, 3, 2, projection=wcs)

    im = plt.imshow(np.log10(pressure_equilibrium),
                    origin='lower',
                    interpolation='none',
                    vmin=vmin, vmax=vmax,
                    cmap='viridis')

    plt.grid()

    ax.coords[0].set_axislabel('RA (J2000)')
    ax.coords[0].set_major_formatter('hh:mm:ss')

    ax.coords[0].display_minor_ticks(True)
    ax.coords[1].display_minor_ticks(True)

    ax.coords[1].set_ticklabel_visible(False)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0, axes_class=matplotlib.axes.Axes)
    plt.colorbar(im, cax=cax, label=r'$\log_{10}(P_\mathrm{DE}~[\mathrm{K~cm^{-3}}])$',
                 orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    # Finally, ratio of the two
    vmin, vmax = -1.5, 1.5

    if galaxy in zoom.keys():

        position = zoom[galaxy]['centre']
        if position is None:
            position = np.asarray(pressure_ratio.shape) / 2
        size = zoom[galaxy]['zoom']

        pressure_ratio = Cutout2D(pressure_ratio,
                                  wcs=wcs_orig,
                                  position=position,
                                  size=size)
        wcs = pressure_ratio.wcs
        pressure_ratio = pressure_ratio.data
    else:
        wcs = copy.deepcopy(wcs_orig)

    ax = plt.subplot(1, 3, 3, projection=wcs)

    im = plt.imshow(pressure_ratio,
                    origin='lower',
                    interpolation='none',
                    vmin=vmin, vmax=vmax,
                    cmap=cmocean.cm.balance)

    plt.grid()

    ax.coords[0].set_axislabel('RA (J2000)')
    ax.coords[1].set_axislabel('Dec (J2000)')

    # ax.coords[0].set_ticklabel(rotation=45, pad=40)
    ax.coords[0].set_major_formatter('hh:mm:ss')

    ax.coords[0].display_minor_ticks(True)
    ax.coords[1].display_minor_ticks(True)

    ax.coords[1].set_ticks_position('lr')
    ax.coords[1].set_ticklabel_position('r')
    ax.coords[1].set_axislabel_position('r')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0, axes_class=matplotlib.axes.Axes)
    plt.colorbar(im, cax=cax, label=r'$\log_{10}(P_\mathrm{turb}/P_\mathrm{DE})$',
                 orientation='horizontal')
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')

    # plt.tight_layout()

    # plt.show()

    plt.savefig(plot_name + '.png', bbox_inches='tight')
    plt.savefig(plot_name + '.pdf', bbox_inches='tight')
    plt.close()

# Now, radial profiles for everything

n_final_galaxies = len(pressure_ratio_radial_values.keys())

plot_name = os.path.join(pressure_plot_dir, 'pressure_comparison_radial')

plt.figure(figsize=(8, 4))
ax = plt.subplot(111)

# colours = itertools.cycle(sns.color_palette('deep'))
colours = iter(plt.cm.viridis(np.linspace(0, 1, len(galaxy_dict.keys()))))

for key in pressure_ratio_radial_values.keys():
    c = next(colours)

    plt.errorbar(pressure_ratio_radial_bins[key],
                 pressure_ratio_radial_values[key][:, 1],
                 yerr=[pressure_ratio_radial_values[key][:, 0], pressure_ratio_radial_values[key][:, 2]],
                 c=c, ls='none', marker='o', label=key.upper())

plt.axhline(0, c='k', ls='--')

plt.legend(loc='center left', frameon=True, fancybox=False, edgecolor='k',
           bbox_to_anchor=(1.0, 0.5))

plt.xlabel(r'$R/R_e$')
plt.ylabel(r'$\log_{10}(P_\mathrm{turb}/P_\mathrm{DE})$')

plt.ylim(-2, 2)

ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

plt.grid()

plt.tight_layout()

# plt.show()

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.close()

print('Complete!')
