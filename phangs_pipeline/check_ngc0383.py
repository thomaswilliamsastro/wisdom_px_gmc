# -*- coding: utf-8 -*-
"""
Check the PHANGS pipeline output versus Eve's cube

@author: Tom Williams
"""

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.convolution import convolve_fft
from radio_beam import Beam
from reproject import reproject_interp
from spectral_cube import SpectralCube

from vars import wisdom_dir

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(wisdom_dir)

# Compare cube-to-cube

orig_cube = SpectralCube.read('new_reduction/test_data/NGC0383_cube.fits')
new_cube = SpectralCube.read('new_reduction/postprocess/ngc0383/ngc0383_12m_co10kms_pbcorr_trimmed_k.fits')

beam_area_orig = orig_cube.header['BMAJ'] * orig_cube.header['BMIN']
beam_area_new = new_cube.header['BMAJ'] * new_cube.header['BMIN']

# Regrid to spatial extend of the original cube (smaller)

if not os.path.exists('new_reduction/test_data/NGC0383_spatial_reproj.fits'):
    new_cube_spatial_reproj = new_cube.reproject(orig_cube.header)
    new_cube_spatial_reproj.write('new_reduction/test_data/NGC0383_spatial_reproj.fits')
else:
    new_cube_spatial_reproj = SpectralCube.read('new_reduction/test_data/NGC0383_spatial_reproj.fits')

# Regrid spectrally
new_cube_spectral_reproj = new_cube_spatial_reproj.spectral_interpolate(orig_cube.spectral_axis + 50 * u.km / u.s)

# Convert the new cube to units of Jy/beam
new_cube_spectral_reproj._data /= new_cube_spectral_reproj.header['JYTOK']

# Convolve the original cube to the new cube beam
beam_to_convolve = Beam(new_cube.beam.major, new_cube.beam.minor)
orig_cube_convolve = orig_cube.convolve_to(beam_to_convolve)

# And account for change of beam size
orig_cube_convolve._data /= beam_area_orig / beam_area_new

new_cube_flat = new_cube_spectral_reproj._data.flatten() * 1e3
orig_cube_flat = orig_cube_convolve._data.flatten() * 1e3

idx = np.where((~np.isnan(new_cube_flat)) & (~np.isnan(orig_cube_flat)))
new_cube_flat = new_cube_flat[idx]
orig_cube_flat = orig_cube_flat[idx]

scatter = np.std(new_cube_flat - orig_cube_flat)
print(np.nanstd(orig_cube._data[0, :, :]) * 1e3)

xmin, xmax = np.nanpercentile(new_cube_flat, [0.1, 99.9])

plt.figure(figsize=(8, 4))
ax = plt.subplot(1, 2, 1)
plt.scatter(orig_cube_flat[::1000], new_cube_flat[::1000], c='k', alpha=0.5)
plt.plot([xmin, xmax], [xmin, xmax], c='r')
plt.xlim(xmin, xmax)
plt.ylim(xmin, xmax)

plt.text(0.05, 0.95, r'$\sigma = %.2f$' % scatter,
         ha='left', va='top', fontsize=16,
         transform=ax.transAxes)

plt.xlabel(r'Original Cube (mJy/beam)')
plt.ylabel('PHANGS Pipeline (mJy/beam)')

plt.subplot(1, 2, 2)
plt.scatter(orig_cube_flat[::1000], (new_cube_flat[::1000] - orig_cube_flat[::1000])/orig_cube_flat[::1000],
            c='k', alpha=0.5)
plt.axhline(0, c='r', ls='--', zorder=99)
plt.xlim(xmin, xmax)
plt.ylim(-5, 5)

plt.xlabel(r'Original Cube (mJy/beam)')
plt.ylabel('(PHANGS Pipeline - Original Cube)/Original Cube')

plt.tight_layout()

# plt.show()

plt.savefig('plots/ngc0383_comparison.png', bbox_inches='tight')

print('Complete!')
