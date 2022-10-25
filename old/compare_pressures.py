# -*- coding: utf-8 -*-
"""
Compare the pressure from the MGE+mom0 to what we see in the data

@author: Tom Williams
"""

import warnings

import cmocean
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.ned import Ned
from astropy.table import Table

from vars import wisdom_dir, plot_dir, galaxies

warnings.simplefilter('ignore')

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(wisdom_dir)

with open('distances.pkl', 'rb') as f:
    distances = pickle.load(f)
with open('resolutions.pkl', 'rb') as f:
    resolutions = pickle.load(f)

if not os.path.exists(os.path.join(plot_dir, 'pressure_comparison')):
    os.makedirs(os.path.join(plot_dir, 'pressure_comparison'))

# galaxies = ['NGC0383', 'NGC0404', 'NGC0524', 'NGC4429', 'NGC4697']

G = 6.67e-11
kb = 1.38e-23

kraj_table = Table.read('literature/Krajnovic2011_Atlas3D_Paper2_TableD1.txt', format='ascii')

# Sources here:
# - 0383 is North+ 2019
# - 0404 is Davis+ 2020
# - 0524 is Smith+ 2019
# - 4429 is Davis+ 2018
# - 4697 is Davis+ 2017

pressure_ratio_radial_bins = {}
pressure_ratio_radial_values = {}

galaxy_info = {'NGC0383': {'pa': 142.2, 'inc': 37.58, 'ml': 2.5,
                           'surf': [12913.045, 3996.8006, 5560.8414, 4962.2927, 2877.9230, 957.88407],
                           'sigma_arcsec': [0.0682436, 0.823495, 1.13042, 2.62962, 4.97714, 12.6523],
                           'qobs': [0.913337, 0.900000, 0.926674, 0.900000, 0.900000, 0.900000],
                           'r25': 16.23
                           },
               'NGC0404': {'pa': 37.2, 'inc': 20, 'ml': 0.66,
                           'surf': [303071.00, 258200.00, 7525.6001, 4395.6001, 2643.0000, 2632.1001, 16183.500,
                                    4477.1001, 3432.1001, 1722.1000, 1353.2100, 948.21002, 506.79401, 204.73599],
                           'sigma_arcsec': [0.0476858, 0.0501783, 0.0726512, 0.0836473, 0.120565, 0.190565, 0.250571,
                                            0.309951, 0.939995, 1.03987, 1.37274, 5.26472, 9.91885, 25.0187],
                           'qobs': [0.97399998, 0.97399998, 0.97399998, 0.97399998, 0.97399998, 0.97399998, 0.97399998,
                                    0.95800000, 0.95800000, 0.95800000, 0.95800000, 0.99699998, 0.99699998, 0.99699998],
                           'r25': 64
                           },
               'NGC0524': {'pa': 39.9, 'inc': 20.6, 'ml': 7,
                           'surf': [10 ** 4.336, 10 ** 3.807, 10 ** 4.431, 10 ** 3.914, 10 ** 3.638, 10 ** 3.530,
                                    10 ** 3.073,
                                    10 ** 2.450, 10 ** 1.832, 10 ** 1.300],
                           'sigma_arcsec': [10 ** -1.762, 10 ** -1.199, 10 ** -0.525, 10 ** -0.037, 10 ** 0.327,
                                            10 ** 0.629,
                                            10 ** 1.082, 10 ** 1.475, 10 ** 1.708, 10 ** 2.132],
                           'qobs': [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
                           'r25': 23.66
                           },
               'NGC4429': {'pa': 93.2, 'inc': 66.8, 'ml': 6.59,
                           'surf': [21146.8, 1268.97, 4555.92, 2848.84, 1438.13, 658.099, 193.243, 26.5331, 6.95278],
                           'sigma_arcsec': [0.144716, 0.547251, 0.749636, 2.15040, 4.82936, 13.4457, 49.4016, 106.119,
                                            106.119],
                           'qobs': [0.675000, 0.950000, 0.400000, 0.700449, 0.780865, 0.615446, 0.400000, 0.400000,
                                    0.950000],
                           'r25': 48.84
                           },
               'NGC4697': {'pa': 246.2, 'inc': 76.1, 'ml': 2.14,
                           'surf': [10 ** 5.579, 10 ** 4.783, 10 ** 4.445, 10 ** 4.239, 10 ** 4.061, 10 ** 3.698,
                                    10 ** 3.314,
                                    10 ** 3.538],
                           'sigma_arcsec': [10 ** -1.281, 10 ** -0.729, 10 ** -0.341, 10 ** -0.006, 10 ** 0.367,
                                            10 ** 0.665,
                                            10 ** 0.809, 10 ** 1.191],
                           'qobs': [0.932, 1, 0.863, 0.726, 0.541, 0.683, 0.318, 0.589],
                           'r25': 39.51
                           },
               'NGC7052': {'pa': 64.3, 'inc': 74.8, 'ml': 3.88,
                           'surf': [10 ** 4.49, 10 ** 3.93, 10 ** 3.67, 10 ** 3.56],
                           'sigma_arcsec': [10 ** -1.76, 10 ** -0.23, 10 ** 0.14, 10 ** 0.6],
                           'qobs': [0.73, 0.77, 0.69, 0.71],
                           'r25': 20.25
                           }
               }

# galaxies = ['NGC0383']

for galaxy in galaxies:

    dist = distances[galaxy]

    pc_conversion_fac = 4.84 * dist  # pc/arcsec

    try:
        pa = galaxy_info[galaxy]['pa']
        inc = galaxy_info[galaxy]['inc']
        ml = np.array(galaxy_info[galaxy]['ml'])
        surf = np.array(galaxy_info[galaxy]['surf'])
        sigma_arcsec = np.array(galaxy_info[galaxy]['sigma_arcsec'])
        qobs = np.array(galaxy_info[galaxy]['qobs'])
        r25 = galaxy_info[galaxy]['r25'] * pc_conversion_fac
    except KeyError:
        continue

    mom0_hdu_file_name = os.path.join('regrids', galaxy + '_native_surf_dens.fits')
    mom0_hdu = fits.open(mom0_hdu_file_name)
    mom0_hdu[0].data[np.isnan(mom0_hdu[0].data)] = 0

    vel_disp_hdu_file_name = os.path.join('regrids', galaxy + '_native_eff_width.fits')
    vel_disp = fits.open(vel_disp_hdu_file_name)[0].data

    # Correct velocity dispersion for channel width

    r_corr = np.loadtxt(os.path.join('regrids', galaxy + '_native_correlation.txt'), usecols=0)

    v_channel = 2
    if galaxy == 'NGC4429':
        v_channel = 3

    k = 0.47 * r_corr - 0.23 * r_corr ** 2 - 0.16 * r_corr ** 3 + 0.43 * r_corr ** 4
    sigma_response = (v_channel / np.sqrt(2 * np.pi)) * (1 + 1.18 * k + 10.4 * k ** 2)
    vel_disp = np.sqrt(vel_disp ** 2 - sigma_response ** 2)

    native_resolution = resolutions[galaxy]
    r_beam = native_resolution / 2

    result_table = Ned.query_object(galaxy)
    ra, dec = result_table['RA'][0], result_table['DEC'][0]
    w = WCS(mom0_hdu_file_name)
    x_cen, y_cen = w.all_world2pix(ra, dec, 1)
    pix_size = np.abs(mom0_hdu[0].header['CDELT1'] * 3600)

    # Now, calculate the pressure from the Sun prescription
    surf_dens = mom0_hdu[0].data

    # Turn the MGE into a stellar mass surface density for each pixel

    yi, xi = np.meshgrid((np.arange(surf_dens.shape[1]) - x_cen) * pix_size * pc_conversion_fac,
                         (np.arange(surf_dens.shape[0]) - y_cen) * pix_size * pc_conversion_fac)

    # Project into galaxy plane

    angle = np.radians(pa)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    x_proj = xi * cos_a + yi * sin_a
    y_proj = - xi * sin_a + yi * cos_a
    x_proj /= np.cos(np.radians(inc))

    r = np.sqrt(x_proj ** 2 + y_proj ** 2)

    sigma = sigma_arcsec * pc_conversion_fac
    qintr2 = qobs ** 2 - np.cos(np.radians(inc)) ** 2
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

    # max_dist = np.nanpercentile(r[~np.isnan(pressure_ratio)], 86)
    max_dist = 0.4 * r25

    radial_bins = np.linspace(0, max_dist, 10)
    bin_width = (radial_bins[1] - radial_bins[0]) / 2

    pressure_ratio_radial_bins[galaxy] = radial_bins / r25

    radial_values = np.zeros([len(radial_bins), 3])

    for i, radial_bin in enumerate(radial_bins):
        idx = np.where((radial_bin - bin_width <= r) & (r <= radial_bin + bin_width))
        radial_values[i, :] = np.nanpercentile(pressure_ratio[idx], [16, 50, 84])
    radial_values[:, 0] = radial_values[:, 1] - radial_values[:, 0]
    radial_values[:, 2] -= radial_values[:, 1]

    pressure_ratio_radial_values[galaxy] = radial_values

    # Plot maps for the pressure ratio

    vmin, vmax = -2, 2

    plot_name = os.path.join(plot_dir, 'pressure_comparison', galaxy + '_map')

    plt.figure()
    plt.imshow(pressure_ratio, origin='lower', vmin=vmin, vmax=vmax, cmap=cmocean.cm.balance)
    plt.colorbar(label=r'$\log10(P_\mathrm{turb}/P_\mathrm{DE})$')

    plt.axis('off')

    plt.title(galaxy)

    # plt.show()

    plt.savefig(plot_name + '.png', bbox_inches='tight')
    plt.savefig(plot_name + '.pdf', bbox_inches='tight')
    plt.close()

# Now, radial profiles for everything

n_final_galaxies = len(pressure_ratio_radial_values.keys())

plot_name = os.path.join(plot_dir, 'pressure_comparison', 'pressure_comparison_radial')

plt.figure(figsize=(8, 4))

colours = iter(plt.cm.rainbow(np.linspace(0, 1, n_final_galaxies)))

for key in pressure_ratio_radial_values.keys():
    c = next(colours)

    plt.errorbar(pressure_ratio_radial_bins[key],
                 pressure_ratio_radial_values[key][:, 1],
                 yerr=[pressure_ratio_radial_values[key][:, 0], pressure_ratio_radial_values[key][:, 2]],
                 c=c, ls='none', marker='o', label=key)

plt.axhline(0, c='k', ls='--')

plt.legend(loc='upper left', frameon=False, bbox_to_anchor=(1.1, 0.9))

plt.xlabel(r'$r/r_{25}$')
plt.ylabel(r'$\log10(P_\mathrm{turb}/P_\mathrm{DE})$')

plt.ylim(-1, 1)

plt.tight_layout()

# plt.show()

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.close()

print('Complete!')
