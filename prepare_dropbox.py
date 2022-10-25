# -*- coding: utf-8 -*-
"""
Prepare folder for Dropbox structure

@author: Tom Williams
"""

import os

from astropy.io import fits
import numpy as np

from vars import galaxies, wisdom_dir

os.chdir(wisdom_dir)

for galaxy in galaxies:

    hdu_in_name = os.path.join('data', galaxy + '.fits')

    hdu = fits.open(hdu_in_name)[0]

    vel_res = '%d' % (np.round(hdu.header['CDELT3'] * 1e-3))

    if vel_res == '3':
        continue

    hdu_out_dir = os.path.join('for_dropbox', galaxy)

    if not os.path.exists(hdu_out_dir):
        os.makedirs(hdu_out_dir)

    hdu_out_name = os.path.join(hdu_out_dir, '%s_%skms_cube.fits' % (galaxy, vel_res))
    hdu.writeto(hdu_out_name, overwrite=True)

print('Complete!')
