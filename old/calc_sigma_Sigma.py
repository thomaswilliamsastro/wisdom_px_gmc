# -*- coding: utf-8 -*-
"""
Calculate the velocity dispersion and gas surface density for the
WISDOM galaxies.

@author: Tom Williams
"""

import copy
import multiprocessing as mp
import os
import pickle
import time
import warnings

import astropy.units as u
import numpy as np
import radio_beam
from astropy.convolution import Gaussian1DKernel
from astropy.io import fits
from spectral_cube import SpectralCube, BooleanArrayMask
from tqdm import tqdm

from vars import wisdom_dir, galaxies, resolutions
from external.sun_cube_tools import find_signal_in_cube, calc_noise_in_cube, calc_channel_corr


def visualisation(input):
    # Add random noise to the cube

    if galaxy == 'NGC3627':
        units = u.K
    else:
        units = u.Jy / u.beam

    cube_masked_noise = cube_masked + np.random.normal(
        loc=0,
        scale=rms_cube.hdu.data) * units

    mom_0_vis = cube_masked_noise.moment(order=0)

    mom_0_vis = mom_0_vis.to(units * u.km/u.s)

    # Convert this through to a map of surface density

    mom_0 = alpha_co * mom_0_vis * jy_beam_conversion

    # And an effective width

    t_peak = cube_masked_noise.max(0)

    eff_width = mom_0_vis / (np.sqrt(2 * np.pi) * t_peak)

    return mom_0, eff_width


warnings.filterwarnings('ignore', append=True)

os.environ['TMPDIR'] = os.path.join(wisdom_dir, 'temp')
os.environ['TEMP'] = os.path.join(wisdom_dir, 'temp')
os.environ['TMP'] = os.path.join(wisdom_dir, 'temp')

# Overwrite switches

overwrite_vel_subtract = False
overwrite_visualise = False

start = time.time()

os.chdir(wisdom_dir)

if not os.path.exists('regrids'):
    os.mkdir('regrids')

if not os.path.exists('regrids/vel_subtract'):
    os.mkdir('regrids/vel_subtract')

fwhm_factor = np.sqrt(8 * np.log(2))

# Read in distance dictionary

with open('distances.pkl', 'rb') as distance_file:
    distance_dict = pickle.load(distance_file)

rms_limit = 3

for galaxy in galaxies:

    print('Processing ' + galaxy)

    distance = distance_dict[galaxy]
    hdu = fits.open('data/' + galaxy + '.fits')[0]

    data_shape = np.min(hdu.data.shape[1:])

    # Set up a cutout of the cube to cut down on regridding time

    try:
        half_size = {
            'NGC4429': 160,
            'NGC0612': 300,
            'NGC0383': 150,
            'NGC4697': 64,
            'NGC4826': 180,
            'NGC5064': 350,
            'NGC0524': 150,
            'NGC1574': 130,
            'NGC7052': 90,
        }[galaxy]
    except KeyError:
        half_size = int(data_shape / 2)

    # Set up min and max channel to use. If not specified use the whole spectral range.

    try:
        chan_min = {
            'NGC4429': 30,
            'NGC0383': 115,
            'NGC0612': 10,
            'NGC4826': 20,
            'NGC5064': 50,
            'NGC0404': 50,
            'NGC0524': 80,
            'NGC1574': 120,
            'NGC7052': 30,
            'NGC3393': 100,
            'NGC0449': 150,
            'NGC3368': 50,
        }[galaxy]
    except KeyError:
        chan_min = 0

    try:
        chan_max = {
            'NGC4429': 305,
            'NGC0383': 415,
            'NGC0612': 425,
            'NGC4826': 200,
            'NGC0404': 140,
            'NGC0524': 275,
            'NGC1574': 350,
            'NGC7052': 500,
            'NGC3393': 320,
            'NGC0449': 350,
            'NGC3368': 360,
        }[galaxy]
    except KeyError:
        chan_max = -1

    # CO line ratio

    try:

        line_ratio = {
            # 'NGC4429':1.06,
            'NGC4429': 0.25,
            'NGC0383': 1,
            'NGC4826': 0.25,
        }[galaxy]

    except KeyError:

        line_ratio = 0.7

    alpha_co = 4.35 / line_ratio

    # Required information in the original cube

    cube_orig = SpectralCube.read('data/' + galaxy + '.fits')
    bmaj_orig = cube_orig.beam.major.to('arcsec')
    bmin_orig = cube_orig.beam.minor.to('arcsec')

    # Allow huge operations on big ol' cubes

    cube_orig.allow_huge_operations = True

    freq = cube_orig.header['RESTFRQ'] * u.Hz
    vel_res = cube_orig.header['CDELT3'] / 1e3 * u.km / u.s

    pix_size_orig = np.abs(cube_orig.header['CDELT1']) * 3600 * u.arcsec

    beam_area_orig = cube_orig.beam.sr.to(u.arcsec ** 2)

    if not os.path.exists('regrids/vel_subtract/' + galaxy + '.fits') or \
            overwrite_vel_subtract:

        # Correct for the primary beam pickup
        beam = fits.open('data/' + galaxy + '_beam.fits')[0]
        cube_orig.hdu.data /= beam.data

        # Pull out the sub-cube

        x_cen = int(np.round(cube_orig.shape[1] / 2))
        y_cen = int(np.round(cube_orig.shape[2] / 2))

        cube_sub = cube_orig[chan_min:chan_max,
                   x_cen - half_size:x_cen + half_size,
                   y_cen - half_size:y_cen + half_size]

        vel_delt = cube_sub.header['CDELT3']
        vel_val = cube_sub.header['CRVAL3']
        vel_pix = cube_sub.header['CRPIX3']

        velocity = np.array([vel_val + (i - (vel_pix - 1)) * vel_delt for i in range(cube_sub.shape[0])])
        velocity /= 1e3

        # Calculate the mean velocity as the systemic

        spectra_integrated = np.array(
            [np.nansum(cube_sub[i, :, :]).value for i in range(cube_sub.shape[0])])

        sys_vel = np.nansum(spectra_integrated * velocity) / np.nansum(spectra_integrated)

        # And subtract the systemic

        velocity -= sys_vel

        # Subtract velocities based on peak of spectrum down line of
        # sight -- Koch et al. (2018) argue this is less susceptible to
        # asymmetry in the wings

        velocity_to_subtract = velocity[cube_sub.argmax(0)]
        velocity_shift = np.round(
            velocity_to_subtract / (vel_delt * 1e-3)).astype(int)

        cube_data = cube_sub.hdu.data.copy()
        cube_shift = np.zeros(cube_sub.shape)

        for i in tqdm(range(cube_shift.shape[1]),
                      desc='Shifting velocities'):
            for j in range(cube_sub.shape[2]):
                cube_shift[:, i, j] = np.roll(cube_data[:, i, j],
                                              -velocity_shift[i, j],
                                              )

        # Write out this cube

        fits.writeto('regrids/vel_subtract/' + galaxy + '.fits',
                     cube_shift, cube_sub.hdu.header,
                     overwrite=True)

    cube_sub = SpectralCube.read('regrids/vel_subtract/' + galaxy + '.fits')

    # Again, allow for huge operations.

    cube_sub.allow_huge_operations = True

    for resolution in resolutions:

        cube_file = 'regrids/' + galaxy + '_' + str(resolution)

        # Calculate the size of the beam we're convolving to in arcsec.

        if isinstance(resolution, str):

            beam_new = (bmaj_orig + bmin_orig) / 2

        else:

            beam_new = (resolution / (4.84 * distance)) * u.arcsec

        if not os.path.exists(cube_file + '.fits'):

            if beam_new > 1.05 * (bmaj_orig + bmin_orig) / 2:

                beam = radio_beam.Beam(
                    major=beam_new,
                    minor=beam_new,
                    pa=0.0 * u.deg,
                )

                cube_new = cube_sub.convolve_to(beam)
                beam_area_new = cube_new.beam.sr.to(u.arcsec ** 2)

                cube_new /= (beam_area_orig / beam_area_new)

            else:

                cube_new = copy.copy(cube_sub)

            if beam_new < bmaj_orig and resolution != 'native':

                # We can't deconvolve, so continue
                continue

            # Downsample the cube to Nyquist sample it

            pix_downsample = int(np.round((beam_new / 3) / pix_size_orig))

            cube_final = cube_new.downsample_axis(
                pix_downsample,
                1,
            )

            cube_final = cube_final.downsample_axis(
                pix_downsample,
                2,
            )

            # Calculate the correlation from the first 10 PPV slices. Make sure we don't have any NaNs lying around.

            mask_array = np.zeros_like(cube_final.hdu.data, dtype=bool)
            mask_array[0:10, :, :] = True
            mask_array[np.isnan(cube_final.hdu.data)] = False

            correlation, p_value = calc_channel_corr(cube_final, mask=mask_array)
            np.savetxt(cube_file + '_correlation.txt', np.c_[correlation, p_value])

            # Mask cube, by calculating RMS cube. Save both out.

            rms_cube = calc_noise_in_cube(cube_final)
            cube_masked = find_signal_in_cube(cube_final, rms_cube, snr_hi=rms_limit)

            cube_masked.write(cube_file + '.fits', overwrite=True)
            rms_cube.write(cube_file + '_rms.fits', overwrite=True)

        else:

            cube_masked = SpectralCube.read(cube_file + '.fits')
            rms_cube = SpectralCube.read(cube_file + '_rms.fits')

        cube_masked.allow_huge_operations = True

        if not os.path.exists(cube_file + '_surf_dens.fits') or \
                overwrite_visualise:

            # Now, we create 1000 visualisations to calculate the error
            # and covariance between the parameters

            n_visualisations = 1000

            # Calculate an initial mom0 just for the header and size

            mom_0 = cube_masked.moment(order=0)
            hdr = mom_0.hdu.header

            mom_0 = np.zeros([mom_0.shape[0],
                              mom_0.shape[1],
                              n_visualisations])
            mom_0[mom_0 == 0] = np.nan

            eff_width = mom_0.copy()

            # And errors

            mom_0_err = np.zeros([mom_0.shape[0],
                                  mom_0.shape[1]])
            eff_width_err = mom_0_err.copy()

            # And covariances

            cov = mom_0_err.copy()

            # Convert the Jy/beam to K through the beam area

            if galaxy == 'NGC3627':
                jy_beam_conversion = 1
            else:
                jy_beam_area = 1 * u.Jy / (cube_masked.beam.sr.to(u.arcsec ** 2))

                jy_beam_conversion = jy_beam_area.to(
                    u.K,
                    equivalencies=u.brightness_temperature(freq))

            # Visualise a number of cubes -- use multiprocessing here for speedy goodness

            pool = mp.Pool(int(mp.cpu_count()*0.75))

            visualised_parameters = list(tqdm(pool.imap(
                visualisation,
                range(n_visualisations)),
                total=n_visualisations, desc='Visualising'))

            pool.close()
            pool.join()

            for i in range(n_visualisations):
                mom_0[:, :, i] = visualised_parameters[i][0]
                eff_width[:, :, i] = visualised_parameters[i][1]

            mom_0_median = np.nanmedian(mom_0, axis=2)
            eff_width_median = np.nanmedian(eff_width, axis=2)

            # Pull errors in log space out from covariance matrix

            for i in range(mom_0.shape[0]):
                for j in range(mom_0.shape[1]):
                    # Only take positive values (else we get NaN'd up!)

                    idx = np.where((mom_0[i, j, :] > 0) & (eff_width[i, j, :] > 0))

                    cov_idx = np.cov(np.log10(mom_0[i, j, :][idx]),
                                     np.log10(eff_width[i, j, :][idx]))

                    mom_0_err[i, j] = np.sqrt(cov_idx[0, 0])
                    eff_width_err[i, j] = np.sqrt(cov_idx[1, 1])

                    cov[i, j] = cov_idx[0, 1]

            surf_dens_limit = alpha_co * np.median(rms_cube) * rms_limit * jy_beam_conversion * vel_res

            # Write out files

            fits.writeto(cube_file + '_surf_dens.fits',
                         mom_0_median,
                         hdr,
                         overwrite=True,
                         )

            fits.writeto(cube_file + '_eff_width.fits',
                         eff_width_median,
                         hdr,
                         overwrite=True,
                         )

            fits.writeto(cube_file + '_surf_dens_err.fits',
                         mom_0_err,
                         hdr,
                         overwrite=True, )

            fits.writeto(cube_file + '_eff_width_err.fits',
                         eff_width_err,
                         hdr,
                         overwrite=True,
                         )

            fits.writeto(cube_file + '_cov.fits',
                         cov,
                         hdr,
                         overwrite=True,
                         )

print('Complete! Took %2.fs' % (time.time() - start))
