import itertools
import multiprocessing as mp
import os
import pickle
import warnings

import corner
import emcee
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.io import fits
from scipy.stats import kendalltau
from spectral_cube import SpectralCube

from external.sun_cube_tools import calc_channel_corr, censoring_function
from vars import wisdom_dir, plot_dir, galaxy_dict, mask, vel_res, co_conv_factors


def collapse_dict(dict_to_collapse):
    dict_collapsed = np.array([])
    for key in dict_to_collapse.keys():
        dict_collapsed = np.append(dict_collapsed, dict_to_collapse[key])

    return dict_collapsed


def odr_fit(theta, x):
    return 10 ** theta[0] * (x / 1e2) ** theta[1]


def lnprob(theta, s2, s0, s0_err):
    lp = lnprior(theta)

    if not np.isfinite(lp):
        return - np.inf

    return lnlike(theta, s2, s0, s0_err)


def lnlike(theta, s2, s0, s0_err):
    beta, a, scatter = theta

    # x = np.array([1, -beta])[np.newaxis]

    model = beta * s2 + a
    total_err = scatter ** 2 + s0_err ** 2
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
sns.set_color_codes()

mp.set_start_method('fork')

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
points_relation_subtracted = {}

target_resolutions = ['60pc', '90pc', '120pc']
surf_dens_lims = [0.8, 12000]
vel_disp_lims = [0.5, 110]

overwrite_corner = False

for target_resolution in target_resolutions:
    vel_disp_dict[target_resolution] = {}
    surf_dens_dict[target_resolution] = {}
    points_relation_subtracted[target_resolution] = {}

# colour = itertools.cycle(sns.color_palette('deep'))
colour = iter(plt.cm.viridis(np.linspace(0, 1, len(galaxy_dict.keys()))))

for galaxy in galaxy_dict.keys():

    c = next(colour)

    co_line = galaxy_dict[galaxy]['co_line']
    antenna_config = galaxy_dict[galaxy]['antenna_config']

    plot_name = os.path.join(sigma_plot_dir, galaxy + '_sigma_Sigma')

    galaxy_target_resolutions = []
    for target_resolution in target_resolutions:

        if target_resolution in galaxy_dict[galaxy]['resolutions']:
            galaxy_target_resolutions.append(target_resolution)

    fig, axes = plt.subplots(figsize=(8 / 3 * len(galaxy_target_resolutions), 4),
                             nrows=1, ncols=len(galaxy_target_resolutions))

    for i, target_resolution in enumerate(galaxy_target_resolutions):

        target_resolution_kpc = float(target_resolution.strip('pc'))

        vel_disp_file_name = os.path.join('new_reduction', 'derived', galaxy,
                                          '%s_%s_%s_%s_%s_%s_ew.fits' % (
                                              galaxy, antenna_config, co_line, vel_res, target_resolution, mask))
        surf_dens_file_name = vel_disp_file_name.replace('_ew', '_mom0')

        vel_disp_err_file_name = vel_disp_file_name.replace('_ew', '_eew')

        vel_disp_hdu = fits.open(vel_disp_file_name)[0]
        vel_disp_err_hdu = fits.open(vel_disp_err_file_name)[0]
        surf_dens_hdu = fits.open(surf_dens_file_name)[0]

        beam_fwhm = surf_dens_hdu.header['BMAJ']
        pix_size = np.abs(surf_dens_hdu.header['CDELT1'])

        pix_per_beam = beam_fwhm / pix_size
        areal_oversampling = np.pi / (4 * np.log(2)) * pix_per_beam ** 2

        co_conv_factor = co_conv_factors[co_line]

        vel_disp = vel_disp_hdu.data
        vel_disp_err = vel_disp_err_hdu.data
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

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            k = 0.47 * r - 0.23 * r ** 2 - 0.16 * r ** 3 + 0.43 * r ** 4
            sigma_resp = vel_res_int / np.sqrt(2 * np.pi) * (1 + 1.8 * k + 10.4 * k ** 2)
            vel_disp = np.sqrt(vel_disp ** 2 - sigma_resp ** 2)

        # Also calculate the completeness limit from the noise cube

        noise_cube_file_name = os.path.join('new_reduction', 'derived', galaxy,
                                            '%s_%s_%s_%s_%s_noise.fits' %
                                            (galaxy, antenna_config, co_line, vel_res, target_resolution))
        noise_cube = fits.open(noise_cube_file_name)[0].data
        noise = np.nanmedian(noise_cube)

        censor_vel_disp = np.linspace(vel_disp_lims[0], vel_disp_lims[1])

        # Kernel rounding errors
        k = k.astype(np.float32)

        censor_surf_dens = censoring_function(censor_vel_disp,
                                              vel_res_int,
                                              noise,
                                              spec_resp_kernel=[k, 1 - 2 * k, k],
                                              snr_crit=3)
        censor_surf_dens *= co_conv_factor

        # Calculate a mask of points that we shouldn't fit to
        censored_real_data = censoring_function(vel_disp,
                                                vel_res_int,
                                                noise,
                                                spec_resp_kernel=[k, 1 - 2 * k, k],
                                                snr_crit=3,
                                                )
        censor_mask = surf_dens > censored_real_data * co_conv_factor

        sampler_filename = os.path.join('samples',
                                        '%s_%s_samples.pkl' % (galaxy, target_resolution))

        if not os.path.exists('samples'):
            os.makedirs('samples')

        if not os.path.exists(sampler_filename):

            ndim, nwalkers = (3, 100)

            pos = []

            initial_beta = 1
            initial_A = 1
            initial_scatter = 0

            for _ in range(nwalkers):
                beta_var = np.abs(np.random.normal(loc=initial_beta, scale=1e-2 * initial_beta))
                A_var = np.abs(np.random.normal(loc=initial_A, scale=1e-2 * initial_A))
                scatter_var = np.abs(np.random.normal(loc=initial_scatter, scale=1e-2))

                pos.append([beta_var, A_var,
                            scatter_var])

            nsteps = 500

            with mp.Pool(int(mp.cpu_count() - 1)) as pool:

                sampler = emcee.EnsembleSampler(nwalkers,
                                                ndim,
                                                lnprob,
                                                args=(np.log10(surf_dens[censor_mask] / 1e2),
                                                      np.log10(vel_disp[censor_mask]),
                                                      0.434 * vel_disp_err[censor_mask] / vel_disp[censor_mask]),
                                                pool=pool)

                # Run the MCMC -- nsteps for nwalkers, but throw away the first half as burn-in

                sampler.run_mcmc(pos, nsteps, progress=True)

            samples = sampler.chain[:, int(nsteps / 2):, :].reshape((-1, ndim))

            with open(sampler_filename, 'wb') as samples_dj:
                pickle.dump(samples, samples_dj)

        else:

            with open(sampler_filename, 'rb') as samples_dj:
                samples = pickle.load(samples_dj)

        percentiles = zip(*np.percentile(samples, [16, 50, 84], axis=0))

        beta_mcmc, A_mcmc, scatter_mcmc = map(lambda percentiles: (percentiles[1],
                                                                   percentiles[2] - percentiles[1],
                                                                   percentiles[1] - percentiles[0]),
                                              percentiles)

        fit_param_file_name = os.path.join('fit_params', '%s_%s.txt' % (galaxy, target_resolution))

        if not os.path.exists('fit_params'):
            os.makedirs('fit_params')

        np.savetxt(fit_param_file_name, np.c_[beta_mcmc, A_mcmc, scatter_mcmc])

        # Corner plot for sanity

        samples_plot_range = [0.995] * samples.shape[1]

        corner_dir = os.path.join(plot_dir, 'sigma_Sigma_corners')

        if not os.path.exists(corner_dir):
            os.makedirs(corner_dir)

        corner_plot_name = os.path.join(corner_dir, '%s_%s_corner' % (galaxy, target_resolution))

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

        # Calculate virialised lines
        surf_dens_vir = np.linspace(surf_dens_lims[0], surf_dens_lims[1], 1000)
        vel_disp_vir = (f * alpha_vir * G / 5 * np.pi / np.log(2)) ** 0.5 * \
                       (target_resolution_kpc / 2) ** 0.5 * surf_dens_vir ** 0.5

        axes[i].scatter(surf_dens, vel_disp, color=c, s=2, rasterized=True)
        axes[i].plot(surf_dens_vir, vel_disp_vir, c='k', ls='--')
        axes[i].axhline(vel_res_int, c='k', ls=':')
        # axes[i].plot(test_mol_sens_limit, test_vel_disp_limit, c='k', ls='--')
        axes[i].plot(censor_surf_dens, censor_vel_disp, c='red', ls='-.')

        # Plot best fit line on with one-sigma errors
        mcmc_median = 10 ** A_mcmc[0] * (surf_dens_vir / 10 ** 2) ** beta_mcmc[0]

        y_to_percentile = []

        for beta, A in samples[:, 0:2][np.random.randint(len(samples), size=250)]:
            y = 10 ** A * (surf_dens_vir / 10 ** 2) ** beta

            y_to_percentile.append(y)

        y_upper = np.percentile(y_to_percentile, 84,
                                axis=0)
        y_lower = np.percentile(y_to_percentile, 16,
                                axis=0)

        axes[i].plot(surf_dens_vir,
                     mcmc_median,
                     c='red',
                     )

        axes[i].fill_between(surf_dens_vir,
                             y_lower, y_upper,
                             facecolor='red',
                             interpolate=True,
                             lw=0.5,
                             edgecolor='red',
                             alpha=0.5,
                             )

        axes[i].set_xscale('log')
        axes[i].set_yscale('log')

        axes[i].set_ylim(vel_disp_lims)
        axes[i].set_xlim(surf_dens_lims)

        locmaj = matplotlib.ticker.LogLocator(base=10, numticks=3)
        axes[i].xaxis.set_major_locator(locmaj)

        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                              numticks=12)
        axes[i].xaxis.set_minor_locator(locmin)
        axes[i].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        axes[i].grid()

        if i == 0:
            axes[i].set_ylabel(r'$\sigma_\mathrm{%s}\,(\mathrm{km\,s^{-1}})$' % target_resolution)
        if i > 0:
            axes[i].tick_params(labelleft=False)
        if i == len(target_resolutions) - 1:
            axes[i].set_ylabel(r'$\sigma_\mathrm{%s}\,(\mathrm{km\,s^{-1}})$' % target_resolution)
            axes[i].tick_params(right=True, labelright=True, which='both')
            axes[i].yaxis.set_label_position('right')

        axes[i].set_xlabel(r'$\Sigma_\mathrm{%s}\,(\mathrm{M}_\odot\,\mathrm{pc}^{-2})$' % target_resolution)

        # Add in the Kendall tau
        nan_idx = np.where(~np.isnan(surf_dens) & ~np.isnan(vel_disp))
        tau, _ = kendalltau(surf_dens[nan_idx], vel_disp[nan_idx])

        axes[i].text(0.05, 0.95, r'$\tau = %.2f$' % tau,
                     ha='left', va='top',
                     transform=axes[i].transAxes,
                     bbox=dict(facecolor='white', edgecolor='k'))

        if target_resolution == target_resolutions[-1]:
            axes[i].text(0.95, 0.05, galaxy.upper(),
                         # fontweight='bold',
                         ha='right', va='bottom',
                         transform=axes[i].transAxes,
                         bbox=dict(facecolor='white', edgecolor='k'))

        points_sub = np.log10(vel_disp[nan_idx]) - (beta_mcmc[0] * np.log10(surf_dens[nan_idx] / 10 **2))
        print('%s %s scatter: %.2f' % (galaxy, target_resolution, np.nanstd(points_sub)))

        # Save values to dict
        vel_disp_dict[target_resolution][galaxy] = vel_disp
        surf_dens_dict[target_resolution][galaxy] = surf_dens
        points_relation_subtracted[target_resolution][galaxy] = points_sub

    # plt.suptitle(galaxy.upper())

    plt.tight_layout()
    plt.subplots_adjust(wspace=0)

    # plt.show()
    plt.savefig(plot_name + '.pdf', bbox_inches='tight', dpi=200)
    plt.savefig(plot_name + '.png', bbox_inches='tight', dpi=200)
    plt.close()

points_relation_subtracted_best_res = []
for galaxy in galaxy_dict.keys():
    if galaxy in points_relation_subtracted[target_resolutions[0]].keys():
        points_relation_subtracted_best_res.extend(list(points_relation_subtracted[target_resolutions[0]][galaxy]))
points_relation_subtracted_best_res = np.array(points_relation_subtracted_best_res)
combined_scatter_best_res = np.nanstd(points_relation_subtracted_best_res)
print(combined_scatter_best_res)

# Complete plot
plot_name = os.path.join(sigma_plot_dir, 'sigma_Sigma_overview')

fig, axes = plt.subplots(figsize=(8 / 3 * len(target_resolutions), 4),
                         nrows=1, ncols=len(target_resolutions))

for target_resolution in target_resolutions:

    total_vel_disp = np.empty(0)
    for galaxy in vel_disp_dict[target_resolution].keys():
        total_vel_disp = np.append(total_vel_disp, vel_disp_dict[target_resolution][galaxy])

    print(np.nanmedian(total_vel_disp))
no

for i, target_resolution in enumerate(target_resolutions):

    # colour = iter(plt.cm.rainbow(np.linspace(0, 1, len(galaxy_dict.keys()))))
    # colour = itertools.cycle(sns.color_palette('deep'))
    colour = iter(plt.cm.viridis(np.linspace(0, 1, len(galaxy_dict.keys()))))

    n_gal = 0

    for galaxy in galaxy_dict.keys():
        c = next(colour)

        if galaxy in vel_disp_dict[target_resolution].keys():
            axes[i].scatter(surf_dens_dict[target_resolution][galaxy],
                            vel_disp_dict[target_resolution][galaxy],
                            color=c,
                            # ls='none',
                            # marker='o',
                            s=2,
                            rasterized=True,
                            # label=galaxy.upper()
                            )

            n_gal += 1

        # Cheeky size for the scatter plot
        axes[i].scatter(-999,
                        -999,
                        color=c,
                        # ls='none',
                        # marker='o',
                        # s=2,
                        rasterized=True,
                        label=galaxy.upper()
                        )

    axes[i].axhline(vel_res_int, c='k', ls=':')

    # Calculate Tau
    total_vel_disp = np.empty(0)
    total_surf_dens = np.empty(0)
    for galaxy in vel_disp_dict[target_resolution].keys():
        total_vel_disp = np.append(total_vel_disp, vel_disp_dict[target_resolution][galaxy])
        total_surf_dens = np.append(total_surf_dens, surf_dens_dict[target_resolution][galaxy])

    total_surf_dens = np.array(total_surf_dens)
    total_vel_disp = np.array(total_vel_disp)

    nan_idx = np.where(~np.isnan(total_surf_dens) & ~np.isnan(total_vel_disp))
    tau, _ = kendalltau(total_surf_dens[nan_idx], total_vel_disp[nan_idx])

    # Calculate virialised lines
    target_resolution_kpc = float(target_resolution.strip('pc'))
    surf_dens_vir = np.linspace(surf_dens_lims[0], surf_dens_lims[1], 1000)
    vel_disp_vir = (f * alpha_vir * G / 5 * np.pi / np.log(2)) ** 0.5 * \
                   (target_resolution_kpc / 2) ** 0.5 * surf_dens_vir ** 0.5

    axes[i].plot(surf_dens_vir, vel_disp_vir, c='k', ls='--')

    axes[i].set_xscale('log')
    axes[i].set_yscale('log')

    axes[i].set_ylim(vel_disp_lims)
    axes[i].set_xlim(surf_dens_lims)

    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=3)
    axes[i].xaxis.set_major_locator(locmaj)

    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                          numticks=12)
    axes[i].xaxis.set_minor_locator(locmin)
    axes[i].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    axes[i].grid()

    if i == 0:
        axes[i].set_ylabel(r'$\sigma\,(\mathrm{km\,s^{-1}})$')
    if i > 0:
        axes[i].tick_params(labelleft=False)
    if i == len(target_resolutions) - 1:
        axes[i].set_ylabel(r'$\sigma\,(\mathrm{km\,s^{-1}})$')
        axes[i].tick_params(right=True, labelright=True, which='both')
        axes[i].yaxis.set_label_position('right')

    axes[i].set_xlabel(r'$\Sigma_\mathrm{%s}\,(\mathrm{M}_\odot\,\mathrm{pc}^{-2})$' % target_resolution)

    plt.text(0.05, 0.95,
             r'$n_{\rm gal}$: %d' % n_gal + '\n' + r'$\tau=%.2f$' % tau,
             ha='left', va='top',
             transform=axes[i].transAxes,
             bbox=dict(facecolor='white', edgecolor='k'),
             )

plt.tight_layout()
plt.subplots_adjust(wspace=0)

lg = plt.legend(loc='center left',
                bbox_to_anchor=(1.35, 0.5),
                frameon=True,
                edgecolor='k',
                framealpha=1,
                fancybox=False,
                scatterpoints=1,
                numpoints=1,
                )

plt.savefig(plot_name + '.pdf', bbox_inches='tight', dpi=200)
plt.savefig(plot_name + '.png', bbox_inches='tight', dpi=200)

plt.close()

print('Complete!')
