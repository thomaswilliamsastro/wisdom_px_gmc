# -*- coding: utf-8 -*-
"""
Setup data for testing

@author: Tom Williams
"""

import os
import numpy as np

from astropy.io import fits

from vars import wisdom_dir

os.chdir(wisdom_dir)

files_to_prep = ['NGC3627', 'KinMS_simcube']

for file_to_prep in files_to_prep:

    hdu = fits.open('data/' + file_to_prep + '.fits')[0]

    beam = np.ones(hdu.data.shape)

    fits.writeto('data/' + file_to_prep + '_beam.fits',
                 beam, hdu.header, overwrite=True)
