# -*- coding: utf-8 -*-
"""
Plot velocity dispersion against H2 surface density

@author: Tom Williams
"""

import warnings

import astropy.units as u
import cmocean
import corner
import dill
import emcee
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pickle
import time
import wget
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.wcs import WCS
from scipy.stats import gaussian_kde, kendalltau
from tqdm import tqdm
from astroquery.ned import Ned


from vars import wisdom_dir, plot_dir, galaxies, resolutions


def odr_fit(theta, x):
    return 10 ** theta[0] * (x / 1e2) ** theta[1]


def lnprob(theta, S2, s0):
    lp = lnprior(theta)

    if not np.isfinite(lp):
        return - np.inf

    return lnlike(theta, S2, s0)


def lnlike(theta, S2, s0):
    beta, A, scatter = theta

    x = np.array([1, -beta])[np.newaxis]

    model = beta * S2 + A
    total_err = np.array([scatter ** 2 + \
                          (float(x * correlation_matrices[i] * x.T)) for i in range(len(S2))])  # /(1+beta**2)
    first_term = (s0 - model) ** 2 / total_err
    second_term = np.log(2 * np.pi * total_err)

    chisq = np.nansum(first_term + second_term)

    return -0.5 * chisq


def lnprior(theta):
    # We only have one prior here -- the scatter must be greater
    # than or equal to 0

    scatter = theta[2]

    if scatter >= 0:
        return 0.0
    return -np.inf


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

# Ignore any RuntimeWarnings in the MCMC
warnings.filterwarnings('ignore', append=True)

os.chdir(wisdom_dir + '/regrids')

fit_param_dir = 'sigma_Sigma_fits'
if not os.path.exists(fit_param_dir):
    os.makedirs(fit_param_dir)

if not os.path.exists('samples'):
    os.mkdir('samples')
if not os.path.exists('gauss_kde_colours'):
    os.mkdir('gauss_kde_colours')

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

point_colour = 'galaxy'
normalise_med_sigma = False

# TODO: Dumb bug in the code means all the fits need to be redone :(

kraj_table = Table.read('../literature/Krajnovic2011_Atlas3D_Paper2_TableD1.txt', format='ascii')
cap_table = Table.read('../literature/Cappellari2013a_Atlas3D_Paper15_Table1.txt', format='ascii')

with open('../distances.pkl', 'rb') as distance_file:
    distance_dict = pickle.load(distance_file)

if __name__ == '__main__':

    with open('../resolutions.pkl', 'rb') as resolution_file:
        resolution_dict = pickle.load(resolution_file)

    resolutions_orig = len(resolutions)

    # Overwrite switches

    overwrite_total_samples = False
    overwrite_total_colours = False

    overwrite_corner = False

    plot_all = True

    # PHANGS best fit lines at various resn (for later)

    phangs_best_fits = {'45': [0.48, 0.66, 0.07], '80': [0.47, 0.85, 0.01], '120': [0.37, 0.85, 0.12]}

    # phangs_best_fits['native'] = [0.48,0.66,0.07]

    full_sample_sigma_gas = {}
    full_sample_sigma_gas_err = {}
    full_sample_vel_disp = {}
    full_sample_vel_disp_err = {}
    full_sample_correlation = {}
    full_sample_colours = {}

    n_points = {}
    n_galaxies = {}

    for resolution in resolutions:
        full_sample_sigma_gas[str(resolution)] = []
        full_sample_sigma_gas_err[str(resolution)] = []
        full_sample_vel_disp[str(resolution)] = []
        full_sample_vel_disp_err[str(resolution)] = []
        full_sample_correlation[str(resolution)] = []
        full_sample_colours[str(resolution)] = []

    for galaxy in galaxies:

        if point_colour == 'radius' and galaxy not in kraj_table['col1']:
            continue

        print('Fitting ' + galaxy)

        dist = distance_dict[galaxy]

        n_rows = int(len(resolutions) / resolutions_orig)
        n_points[galaxy] = {}

        plt.figure(figsize=(14,
                            6 * n_rows),
                   dpi=300)

        plt.suptitle(galaxy)

        gs = gridspec.GridSpec(n_rows, resolutions_orig)
        gs.update(wspace=0, hspace=0)

        subplot_row = 0
        subplot_idx = 0

        native_resolution = resolution_dict[galaxy]

        for resolution in resolutions:

            # Skip all this if the native resolution is worse
            # than the resolution we're trying to fit

            if resolution not in ['native']:
                if 1.05 * resolution_dict[galaxy] > resolution:
                    print('Native resolution poorer than resolution trying to fit -- skipping')

                    # AT SOME POINT, TIDY UP THE PLOTS HERE HEY

                    continue

            if resolution not in ['native']:
                resolution_pc = resolution * u.pc
            else:
                resolution_pc = native_resolution * u.pc

            inv_b_to_alphavir = (5.77 * (80. / resolution_pc) * (u.Msun / u.pc ** 2 / (u.km / u.s) ** 2))

            # Read in the velocity dispersion and gas surface density
            # maps

            vel_disp_hdu = fits.open(
                galaxy + '_' + str(resolution) + '_eff_width.fits',
            )[0]
            sigma_gas_hdu = fits.open(
                galaxy + '_' + str(resolution) + '_surf_dens.fits',
            )[0]

            # Also read in the error maps

            vel_disp_hdu_err = fits.open(
                galaxy + '_' + str(resolution) + '_eff_width_err.fits',
            )[0]
            sigma_gas_err_hdu = fits.open(
                galaxy + '_' + str(resolution) + '_surf_dens_err.fits',
            )[0]

            # And the correlation map

            correlation_hdu = fits.open(
                galaxy + '_' + str(resolution) + '_cov.fits',
            )[0]

            # If we're plotting by galactocentric radius, include that here

            if point_colour in ['radius', 'proj_dist']:

                result_table = Ned.query_object(galaxy)
                ra, dec = result_table['RA'][0], result_table['DEC'][0]

                w = WCS(sigma_gas_hdu)
                pix_size = np.abs(sigma_gas_hdu.header['CDELT1']) * 3600

                x_cen, y_cen = w.all_world2pix(ra, dec, 1)

                xi, yi = np.meshgrid((np.arange(sigma_gas_hdu.data.shape[1]) - x_cen),
                                     (np.arange(sigma_gas_hdu.data.shape[0]) - y_cen))

                if point_colour == 'radius':

                    kraj_row = kraj_table[kraj_table['col1'] == galaxy]
                    cap_row = cap_table[cap_table['col1'] == galaxy]

                    # Pull out: inclination, PA, effective radius

                    pa = float(kraj_row['col6'][0])
                    inc = float(cap_row['col5'][0])
                    r_e = 10**float(cap_row['col9'][0])/np.cos(np.radians(inc))

                    r_e = r_e / 3600 * np.pi / 180 * dist * 1e3

                    # Convert these positions to physical positions (kpc), accounting for inclination and rotation

                    xi *= pix_size / 3600 * np.pi / 180 * dist * 1e3
                    yi *= pix_size / 3600 * np.pi / 180 * dist * 1e3

                    angle = pa * np.pi / 180

                    cos_a, sin_a = np.cos(angle), np.sin(angle)

                    x_rot = cos_a * xi + sin_a * yi
                    y_rot = -sin_a * xi + cos_a * yi

                    # Account for inclination

                    y_rot /= np.cos(inc * np.pi / 180)

                    x_rot /= r_e
                    y_rot /= r_e

                elif point_colour == 'proj_dist':

                    x_rot = xi * pix_size / 3600 * np.pi / 180 * dist * 1e6
                    y_rot = yi * pix_size / 3600 * np.pi / 180 * dist * 1e6

                else:

                    raise Warning('point colour %s not recognised!' % point_colour)

                r = np.sqrt(x_rot**2 + y_rot**2)
                r = r.flatten()

            # Flatten and get rid of NaNs

            vel_disp = vel_disp_hdu.data.flatten()
            sigma_gas = sigma_gas_hdu.data.flatten()

            vel_disp_err = vel_disp_hdu_err.data.flatten()
            sigma_gas_err = sigma_gas_err_hdu.data.flatten()

            correlation = correlation_hdu.data.flatten()

            idx = np.where((np.isnan(vel_disp) == False) & (np.isinf(vel_disp) == False) & \
                           (np.isnan(sigma_gas) == False) & (np.isinf(sigma_gas) == False))

            vel_disp = vel_disp[idx]
            sigma_gas = sigma_gas[idx]

            vel_disp_err = vel_disp_err[idx]
            sigma_gas_err = sigma_gas_err[idx]

            correlation = correlation[idx]

            if point_colour in ['radius', 'proj_dist']:
                r = r[idx]

            idx = np.where((vel_disp > 0) & (sigma_gas > 0))

            vel_disp = vel_disp[idx]
            sigma_gas = sigma_gas[idx]

            vel_disp_err = vel_disp_err[idx]
            sigma_gas_err = sigma_gas_err[idx]

            correlation = correlation[idx]

            # Correct the velocity dispersions given the correlation

            r_corr = np.loadtxt(galaxy + '_' + str(resolution) + '_correlation.txt', usecols=0)

            v_channel = 2
            if galaxy == 'NGC4429':
                v_channel = 3

            k = 0.47 * r_corr - 0.23 * r_corr ** 2 - 0.16 * r_corr ** 3 + 0.43 * r_corr ** 4
            sigma_response = (v_channel / np.sqrt(2 * np.pi)) * (1 + 1.18 * k + 10.4 * k ** 2)
            vel_disp = np.sqrt(vel_disp ** 2 - sigma_response ** 2)

            idx = np.where(~np.isnan(vel_disp))

            vel_disp = vel_disp[idx]
            sigma_gas = sigma_gas[idx]

            vel_disp_err = vel_disp_err[idx]
            sigma_gas_err = sigma_gas_err[idx]

            correlation = correlation[idx]

            if point_colour in ['radius', 'proj_dist']:
                r = r[idx]

            if len(sigma_gas) == 0:
                print('No acceptable velocity dispersions -- skipping')
                continue

            try:
                n_galaxies[resolution] += 1
            except KeyError:
                n_galaxies[resolution] = 1

            if normalise_med_sigma:

                med_sigma = np.nanmedian(sigma_gas)

                sigma_gas /= med_sigma
                sigma_gas_err /= med_sigma

            if point_colour in ['radius', 'proj_dist']:
                r = r[idx]

            # Concatenate the whole sample for later. It doesn't make sense to use
            # native resolutions as they vary wildly

            if resolution not in ['native', 'native_vel_subtract']:
                full_sample_sigma_gas[str(resolution)].extend(sigma_gas)
                full_sample_sigma_gas_err[str(resolution)].extend(sigma_gas_err)
                full_sample_vel_disp[str(resolution)].extend(vel_disp)
                full_sample_vel_disp_err[str(resolution)].extend(vel_disp_err)
                full_sample_correlation[str(resolution)].extend(correlation)

                n_points[galaxy][resolution] = len(sigma_gas)

            # Create correlation matrices for each pixel

            correlation_matrices = []

            for i in range(len(sigma_gas)):
                correlation_matrix = np.matrix([[sigma_gas_err[i] ** 2, correlation[i]],
                                                [correlation[i], vel_disp_err[i] ** 2]])

                correlation_matrices.append(correlation_matrix)

            if point_colour == 'gauss_kde':

                # Colour points by Gaussian KDE in ~log~ space

                if not os.path.exists('gauss_kde_colours/' + galaxy + '_' + str(resolution) + '_colours.txt'):

                    # Because for large numbers of points this becomes very slow, save out if they don't
                    # already exist

                    xy = np.vstack([np.log10(sigma_gas), np.log10(vel_disp)])
                    z = gaussian_kde(xy)(xy)

                    np.savetxt('gauss_kde_colours/' + galaxy + '_' + str(resolution) + '_colours.txt', z)

                else:

                    z = np.loadtxt('gauss_kde_colours/' + galaxy + '_' + str(resolution) + '_colours.txt')

                # Sort the points by density, so that the densest points are plotted last
                idx = z.argsort()
                sigma_gas, vel_disp, z = sigma_gas[idx], vel_disp[idx], z[idx]

                sigma_gas_err, vel_disp_err = sigma_gas_err[idx], vel_disp_err[idx]
                vmin, vmax = None, None
                cmap = cmocean.cm.haline

            elif point_colour == 'radius':

                z = r
                vmin, vmax = 0, 1
                cmap = cmocean.cm.haline_r

                full_sample_colours[str(resolution)].extend(r)

            elif point_colour == 'proj_dist':

                z = r

                idx = z.argsort()
                z, sigma_gas, vel_disp, sigma_gas_err, vel_disp_err = z[idx], sigma_gas[idx], vel_disp[idx], \
                                                                      sigma_gas_err[idx], vel_disp_err[idx]

                vmin, vmax = np.nanpercentile(z, 1), np.nanpercentile(z, 99)
                cmap = cmocean.cm.haline_r

                full_sample_colours[str(resolution)].extend(r)

            else:

                z = 'k'
                vmin, vmax = None, None
                cmap = None

            if galaxy == 'KinMS_simcube':
                xmin, xmax = 10 ** -2.7, 10 ** 1.3
            else:
                xmin, xmax = 10 ** -0.7, 10 ** 4.3
            vir_x = np.linspace(xmin, xmax, 1000)

            ax = plt.subplot(gs[subplot_row,
                                subplot_idx])

            plt.scatter(sigma_gas, vel_disp,
                        c=z, marker='o',
                        lw=0,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        rasterized=True)

            if point_colour == 'radius' and resolution == resolutions[-1]:
                plt.colorbar(pad=0, label=r'$r/r_e$')

            if point_colour == 'proj_dist' and resolution == resolutions[-1]:
                plt.colorbar(pad=0, label=r'$r$ (pc)')

            # Fit the line (MCMC)

            # If we don't already have samples, set up the MCMC

            if not os.path.exists('samples/' + galaxy + '_' + str(resolution) + '_samples.hkl'):

                ndim, nwalkers = (3, 100)

                pos = []

                initial_beta = 1
                initial_A = 1
                initial_scatter = 0

                for i in range(nwalkers):
                    beta_var = np.abs(np.random.normal(loc=initial_beta, scale=1e-2 * initial_beta))
                    A_var = np.abs(np.random.normal(loc=initial_A, scale=1e-2 * initial_A))
                    scatter_var = np.abs(np.random.normal(loc=initial_scatter, scale=1e-2))

                    pos.append([beta_var, A_var,
                                scatter_var])

                mcmc_start = time.time()
                nsteps = 500

                pool = mp.Pool(int(mp.cpu_count()))

                sampler = emcee.EnsembleSampler(nwalkers,
                                                ndim,
                                                lnprob,
                                                args=(np.log10(sigma_gas / 1e2),
                                                      np.log10(vel_disp)),
                                                pool=pool)

                # Run the MCMC -- nsteps for nwalkers, but throw away the first half as burn-in

                sampler.run_mcmc(pos, nsteps, progress=True)

                pool.close()
                pool.join()

                samples = sampler.chain[:, int(nsteps / 2):, :].reshape((-1, ndim))

                with open('samples/' + galaxy + '_' + str(resolution) + '_samples.hkl', 'wb') as samples_dj:
                    dill.dump(samples, samples_dj)

            else:

                with open('samples/' + galaxy + '_' + str(resolution) + '_samples.hkl', 'rb') as samples_dj:
                    samples = dill.load(samples_dj)

            percentiles = zip(*np.percentile(samples, [16, 50, 84], axis=0))

            beta_mcmc, A_mcmc, scatter_mcmc = map(lambda percentiles: (percentiles[1],
                                                                       percentiles[2] - percentiles[1],
                                                                       percentiles[1] - percentiles[0]),
                                                  percentiles)

            fit_param_file_name = os.path.join(fit_param_dir, galaxy + '_' + str(resolution) + '.txt')

            np.savetxt(fit_param_file_name, np.c_[beta_mcmc, A_mcmc, scatter_mcmc])

            # Corner plot for sanity

            samples_plot_range = [0.995] * samples.shape[1]

            corner_dir = os.path.join(plot_dir, 'sigma_Sigma_corners')

            if not os.path.exists(corner_dir):
                os.makedirs(corner_dir)

            corner_plot_name = os.path.join(corner_dir, galaxy + '_' + str(resolution) + '_corner')

            if not os.path.exists(corner_plot_name + '.pdf') or overwrite_corner:

                fig = corner.corner(samples,
                                    labels=[r'$\beta$', 'A', r'$\Delta_\mathregular{int}$'],
                                    quantiles=[0.16, 0.84],
                                    show_titles=True,
                                    truths=[beta_mcmc[0], A_mcmc[0], scatter_mcmc[0]],
                                    truth_color='k',
                                    range=samples_plot_range,
                                    title_fmt='.3f')
                fig.savefig(corner_plot_name + '.pdf', bbox_inches='tight')
                fig.savefig(corner_plot_name + '.png', bbox_inches='tight')

                plt.close(fig)

            # Plot this best fit line on, as well as 1-sigma errors

            mcmc_median = 10 ** A_mcmc[0] * (vir_x / 10 ** 2) ** beta_mcmc[0]

            y_to_percentile = []

            for beta, A in samples[:, 0:2][np.random.randint(len(samples), size=250)]:
                y = 10 ** A * (vir_x / 10 ** 2) ** beta

                y_to_percentile.append(y)

            y_upper = np.percentile(y_to_percentile, 84,
                                    axis=0)
            y_lower = np.percentile(y_to_percentile, 16,
                                    axis=0)

            plt.plot(vir_x, mcmc_median, c='red')

            plt.fill_between(vir_x, y_lower, y_upper,
                             facecolor='red', interpolate=True, lw=0.5,
                             edgecolor='red', alpha=0.5)

            #         plt.text(0.05,0.95,r'$\beta$ = $%.2f^{+%.2f}_{-%.2f}$,' % beta_mcmc+' '
            #                  r'A = $%.2f^{+%.2f}_{-%.2f}$,' % A_mcmc+' '
            #                  r'$\Delta_\mathregular{int}$ = $%.2f^{+%.2f}_{-%.2f}$,' % scatter_mcmc,
            #                  horizontalalignment='left',
            #                  verticalalignment='top',
            #                  fontsize=14,
            #                  color='red',
            #                  transform=ax.transAxes)

            try:

                beta_phangs, A_phangs, scat_phangs = phangs_best_fits[str(resolution)]

                # plt.plot(vir_x, 10 ** (beta_phangs * np.log10((vir_x / 10 ** 2)) + A_phangs),
                #          c='k', ls='--',
                #          lw=2)
                plt.plot(vir_x, (vir_x / inv_b_to_alphavir.value) ** 0.5,
                         c='k', ls='--',
                         lw=2)


            #             plt.text(0.05,0.87,r'$\beta$ = %.2f, A = %.2f, $\Delta_\mathregular{int}$ = %.2f'  % (beta_phangs,
            #                                                                                                   A_phangs,
            #                                                                                                   scat_phangs),
            #                      horizontalalignment='left',
            #                      verticalalignment='top',
            #                      fontsize=14,
            #                      color='k',
            #                      transform=ax.transAxes)

            except KeyError:

                pass

            # Plot on the limit of the velocity resolution

            plt.axhline(2, c='k',
                        ls=':',
                        lw=2)

            # Put on the Kendall tau

            tau_value, _ = kendalltau(sigma_gas, vel_disp)

            plt.text(0.05, 0.95, r'$\tau$ : %.2f' % tau_value,
                     horizontalalignment='left',
                     verticalalignment='top',
                     fontsize=14,
                     color='black',
                     transform=ax.transAxes)

            plt.xscale('log')
            plt.yscale('log')

            plt.xlim([xmin, xmax])
            plt.ylim([10 ** -0.3, 10 ** 2])

            if subplot_idx == 0:

                plt.ylabel(r'$\sigma$ (km s$^{-1}$)')

            else:

                ax.set_yticklabels([])

            if subplot_idx == 0 and subplot_row == n_rows - 1:
                plt.ylabel(r'$\sigma_\mathregular{corr}$ (km s$^{-1}$)')

            if subplot_row < n_rows - 1:
                ax.set_xticklabels([])

            if resolution != 'native' and subplot_row == n_rows - 1:
                plt.xlabel(r'$\Sigma_\mathregular{' + str(resolution) + 'pc}$ (M$_\odot$ pc$^{-2}$)')

            if resolution == 'native' and subplot_row == n_rows - 1:
                plt.xlabel(r'$\Sigma_\mathregular{' + str(int(round(native_resolution))) + 'pc}$ (M$_\odot$ pc$^{-2}$)')

            subplot_idx += 1

            if subplot_idx >= resolutions_orig:
                subplot_idx = 0
                subplot_row += 1

            del vel_disp, vel_disp_hdu, sigma_gas, sigma_gas_hdu
            del vel_disp_err, vel_disp_hdu_err, sigma_gas_err, sigma_gas_err_hdu

        #     plt.show()

        plot_full_dir = os.path.join(plot_dir, 'sigma_Sigma_' + point_colour)

        if normalise_med_sigma:
            plot_full_dir += '_normalised'

        if not os.path.exists(plot_full_dir):
            os.makedirs(plot_full_dir)

        plot_name = os.path.join(plot_full_dir, galaxy + '_vel_disp_sigma_gas_' + point_colour)

        plt.savefig(plot_name + '.png', bbox_inches='tight')
        plt.savefig(plot_name + '.pdf', bbox_inches='tight')

        plt.close()

    plt.close('all')

    if plot_all:

        # And the fit for the entire galaxy sample

        plt.figure(figsize=(10,
                            6),
                   dpi=300)
        gs = gridspec.GridSpec(1, resolutions_orig - 1)
        gs.update(wspace=0, hspace=0)

        subplot_idx = 0
        subplot_row = 0

        print('Fitting full sample')

        for resolution in resolutions:

            if resolution not in ['native',
                                  'native_vel_subtract']:

                # Colour points by Gaussian KDE in ~log~ space

                sigma_gas = np.array(full_sample_sigma_gas[str(resolution)])
                sigma_gas_err = np.array(full_sample_sigma_gas_err[str(resolution)])
                vel_disp = np.array(full_sample_vel_disp[str(resolution)])
                vel_disp_err = np.array(full_sample_vel_disp_err[str(resolution)])
                correlation = np.array(full_sample_correlation[str(resolution)])

                # Create correlation matrices for each pixel

                correlation_matrices = []

                for i in range(len(sigma_gas)):
                    correlation_matrix = np.matrix([[sigma_gas_err[i] ** 2, correlation[i]],
                                                    [correlation[i], vel_disp_err[i] ** 2]])

                    correlation_matrices.append(correlation_matrix)

                # Fit the line (MCMC)

                # If we don't already have samples, set up the MCMC

                if not os.path.exists(
                        'samples/full_sample_' + str(resolution) + '_samples.hkl') or overwrite_total_samples:

                    ndim, nwalkers = (3, 100)

                    pos = []

                    initial_beta = 1
                    initial_A = 1
                    initial_scatter = 0

                    for i in range(nwalkers):
                        beta_var = np.abs(np.random.normal(loc=initial_beta, scale=1e-2 * initial_beta))
                        A_var = np.abs(np.random.normal(loc=initial_A, scale=1e-2 * initial_A))
                        scatter_var = np.abs(np.random.normal(loc=initial_scatter, scale=1e-2))

                        pos.append([beta_var, A_var,
                                    scatter_var])

                    mcmc_start = time.time()
                    nsteps = 500

                    pool = mp.Pool(mp.cpu_count())

                    sampler = emcee.EnsembleSampler(nwalkers,
                                                    ndim,
                                                    lnprob,
                                                    args=(np.log10(sigma_gas / 1e2),
                                                          np.log10(vel_disp)),
                                                    pool=pool)

                    # Run the MCMC -- 500 steps for the walkers, but throw away
                    # the first 250 in case they're junk

                    for i, result in tqdm(enumerate(sampler.sample(pos,
                                                                   iterations=nsteps)),
                                          total=nsteps,
                                          desc='Fitting'):
                        pos, probability, state = result

                    pool.close()
                    pool.join()

                    samples = sampler.chain[:, 250:, :].reshape((-1, ndim))

                    with open('samples/full_sample_' + str(resolution) + '_samples.hkl', 'wb') as samples_dj:
                        dill.dump(samples, samples_dj)

                else:

                    with open('samples/full_sample_' + str(resolution) + '_samples.hkl', 'rb') as samples_dj:
                        samples = dill.load(samples_dj)

                percentiles = zip(*np.percentile(samples, [16, 50, 84], axis=0))

                beta_mcmc, A_mcmc, scatter_mcmc = map(lambda percentiles: (percentiles[1],
                                                                           percentiles[2] - percentiles[1],
                                                                           percentiles[1] - percentiles[0]),
                                                      percentiles)

                # Corner plot for sanity

                samples_plot_range = [0.995] * samples.shape[1]

                corner_dir = os.path.join(plot_dir, 'sigma_Sigma_corners')

                if not os.path.exists(corner_dir):
                    os.makedirs(corner_dir)

                corner_plot_name = os.path.join(corner_dir, 'full_sample_' + str(resolution) + '_corner')

                if not os.path.exists(corner_plot_name + '.pdf') or overwrite_corner:

                    fig = corner.corner(samples,
                                        labels=[r'$\beta$', 'A', r'$\Delta_\mathregular{int}$'],
                                        quantiles=[0.16, 0.84],
                                        show_titles=True,
                                        truths=[beta_mcmc[0], A_mcmc[0], scatter_mcmc[0]],
                                        truth_color='k',
                                        range=samples_plot_range,
                                        title_fmt='.3f')
                    fig.savefig(corner_plot_name + '.pdf',
                                bbox_inches='tight')
                    fig.savefig(corner_plot_name + '.png',
                                bbox_inches='tight')

                    plt.close(fig)

                # Plot this best fit line on, as well as 1-sigma errors

                mcmc_median = 10 ** A_mcmc[0] * (vir_x / 10 ** 2) ** beta_mcmc[0]

                y_to_percentile = []

                for beta, A in samples[:, 0:2][np.random.randint(len(samples), size=250)]:
                    y = 10 ** A * (vir_x / 10 ** 2) ** beta

                    y_to_percentile.append(y)

                y_upper = np.percentile(y_to_percentile, 84,
                                        axis=0)
                y_lower = np.percentile(y_to_percentile, 16,
                                        axis=0)

                ax = plt.subplot(gs[subplot_row,
                                    subplot_idx])

                if point_colour == 'gauss_kde':

                    if not os.path.exists('gauss_kde_colours/full_sample_' + str(resolution) + '_colours.txt') \
                            or overwrite_total_colours:

                        xy = np.vstack([np.log10(sigma_gas), np.log10(vel_disp)])
                        z = gaussian_kde(xy)(xy)

                        np.savetxt('gauss_kde_colours/full_sample_' + str(resolution) + '_colours.txt', z)

                    else:

                        z = np.loadtxt('gauss_kde_colours/full_sample_' + str(resolution) + '_colours.txt')

                    # Sort the points by density, so that the densest points are plotted last
                    idx = z.argsort()
                    sigma_gas, vel_disp, z = sigma_gas[idx], vel_disp[idx], z[idx]
                    cmap = cmocean.cm.haline
                    vmin, vmax = None, None
                    plot_alpha = 1

                elif point_colour == 'galaxy':

                    colours = iter(plt.cm.rainbow(np.linspace(0, 1, len(galaxies))))
                    cmap = 'rainbow'
                    vmin, vmax = None, None
                    plot_alpha = 0.5

                    z = []

                    for galaxy in galaxies:
                        c = next(colours)

                        if resolution in n_points[galaxy].keys():
                            colour_labels = [c] * n_points[galaxy][resolution]
                            z.extend(colour_labels)

                        if resolution == resolutions[-1]:
                            plt.scatter([-1], [-1], c=[c], label=galaxy)

                elif point_colour == 'morphology':

                    z = []
                    plot_alpha = 0.5
                    vmin, vmax = None, None
                    cmap = cmocean.cm.haline

                    for galaxy in galaxies:

                        if resolution in n_points[galaxy].keys():

                            # Query HYPERLEDA to pull out a Hubble type

                            if os.path.exists('fG.cgi'):
                                os.remove('fG.cgi')
                            hyperleda_query = wget.download('http://leda.univ-lyon1.fr/fG.cgi?n=a102&c=o&o=' + galaxy +
                                                            '&a=htab')

                            table = ascii.read('fG.cgi', format='html')
                            classification = float(table['type'][0].split('(')[0])

                            if classification <= 0:
                                c = 'r'
                            else:
                                c = 'b'

                            z.extend([c] * n_points[galaxy][resolution])

                elif point_colour == 'radius':

                    z = full_sample_colours[str(resolution)]
                    vmin, vmax = 0, 1
                    cmap = cmocean.cm.haline_r
                    plot_alpha = 1

                elif point_colour == 'proj_dist':

                    z = np.array(full_sample_colours[str(resolution)])

                    idx = z.argsort()
                    z, sigma_gas, vel_disp = z[idx], sigma_gas[idx], vel_disp[idx]

                    vmin, vmax = np.nanpercentile(z, 1), np.nanpercentile(z, 99)
                    cmap = cmocean.cm.haline_r
                    plot_alpha = 1

                else:

                    raise Warning('point_colour %s not understood' % point_colour)

                xmin, xmax = 10 ** -0.7, 10 ** 4.3
                vir_x = np.linspace(xmin, xmax, 1000)

                plt.scatter(sigma_gas, vel_disp,
                            c=z, marker='o',
                            lw=0, cmap=cmap, alpha=plot_alpha, vmin=vmin, vmax=vmax,
                            rasterized=True)

                if point_colour == 'radius' and resolution == resolutions[-1]:
                    plt.colorbar(pad=0, label=r'$r/r_e$')

                if point_colour == 'proj_dist' and resolution == resolutions[-1]:
                    plt.colorbar(pad=0, label=r'$r$ (pc)')

                # Plot on the best fit line

                # plt.plot(vir_x, mcmc_median, c='red')

                # plt.fill_between(vir_x, y_lower, y_upper,
                #                  facecolor='red', interpolate=True, lw=0.5,
                #                  edgecolor='red', alpha=0.5)

                # Plot on the limit of the velocity resolution

                plt.axhline(2, c='k',
                            ls=':',
                            lw=2)

                # Put on the Kendall tau

                tau_value, _ = kendalltau(sigma_gas, vel_disp)

                plt.text(0.05, 0.95, '$\\tau$ : %.2f\n$N_\mathregular{gal}=%d$' % (tau_value, n_galaxies[resolution]),
                         horizontalalignment='left',
                         verticalalignment='top',
                         fontsize=14,
                         color='black',
                         transform=ax.transAxes)

                plt.xscale('log')
                plt.yscale('log')

                plt.xlim([xmin, xmax])
                plt.ylim([10 ** -0.3, 10 ** 2])

                # Strip out the _vel subtract if required

                # resolution = resolution.split('_vel_subtract')[0]

                resolution_pc = resolution * u.pc
                inv_b_to_alphavir = (5.77 * (80. / resolution_pc) * (u.Msun / u.pc ** 2 / (u.km / u.s) ** 2))

                try:

                    beta_phangs, A_phangs, scat_phangs = phangs_best_fits[str(resolution)]

                    # plt.plot(vir_x, 10 ** (beta_phangs * np.log10((vir_x / 10 ** 2)) + A_phangs),
                    #          c='k', ls='--',
                    #          lw=2)
                    plt.plot(vir_x, (vir_x / inv_b_to_alphavir.value) ** 0.5,
                             c='k', ls='--',
                             lw=2)

                except KeyError:

                    pass

                if subplot_idx == 0:

                    plt.ylabel(r'$\sigma$ (km s$^{-1}$)')

                else:

                    ax.set_yticklabels([])

                if subplot_idx == 0 and subplot_row == n_rows - 1:
                    plt.ylabel(r'$\sigma_\mathregular{corr}$ (km s$^{-1}$)')

                if subplot_row < n_rows - 1:
                    ax.set_xticklabels([])

                if resolution != 'native' and subplot_row == n_rows - 1:
                    plt.xlabel(r'$\Sigma_\mathregular{' + str(resolution) + 'pc}$ (M$_\odot$ pc$^{-2}$)')

                if resolution == 'native' and subplot_row == n_rows - 1:
                    plt.xlabel(r'$\Sigma_\mathregular{' + str(native_resolution) + 'pc}$ (M$_\odot$ pc$^{-2}$)')

                subplot_idx += 1

                if subplot_idx >= resolutions_orig:
                    subplot_idx = 0
                    subplot_row += 1

        if point_colour == 'galaxy':
            plt.legend(bbox_to_anchor=(1.1, 1))

        # plt.show()

        plot_full_dir = os.path.join(plot_dir, 'sigma_Sigma_' + point_colour)

        if normalise_med_sigma:
            plot_full_dir += '_normalised'

        if not os.path.exists(plot_full_dir):
            os.makedirs(plot_full_dir)

        plot_name = os.path.join(plot_full_dir, 'full_sample_vel_disp_sigma_gas_' + point_colour)

        plt.savefig(plot_name + '.png', bbox_inches='tight', dpi=150)
        plt.savefig(plot_name + '.pdf', bbox_inches='tight')

        plt.close()

    print('Complete!')
