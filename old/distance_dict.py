# -*- coding: utf-8 -*-
"""
Save dictionary of distances for easy import later

@author: Tom Williams
"""

import numpy as np
import os
import pickle
from astropy.io import fits

from vars import wisdom_dir

os.chdir(wisdom_dir)

# Distances

distance = {'NGC4429': 16.5,  # 1012.1551
            'NGC0383': 66.6,  # 0012376
            'NGC4697': 11.4,  # 1012.1551
            'NGC4826': 7.36,  # 1012.1551
            'NGC5064': 36.7,  # Onishi subm.
            'NGC0404': 3.06,  # 0204507
            'NGC0524': 23.3,  # 1012.1551
            'NGC4501': 15.3,  # 1012.1551
            'NGC1194': 56.9,  # 1107.1237
            'M33': 0.84,  # For testing
            'NGC1574': 16.7,
            'NGC5084': 41.3,
            'NGC7052': 51.6,
            'NGC3393': 52.7,
            'NGC0612': 128.2,
            'NGC0449': 71.8,
            'NGC3368': 10.1,
            'NGC3627': 11.32,
            'KinMS_simcube': 10,
            }

with open('distances.pkl', 'wb') as distance_file:
    pickle.dump(distance, distance_file)

# And corresponding native resolutions

native_resoln = {}

for filename in distance:
    hdu = fits.open('data/' + filename + '.fits')[0]

    beam_ave = (hdu.header['BMAJ'] + hdu.header['BMIN']) / 2

    beam_ave *= np.pi / 180
    resolution = distance[filename] * beam_ave * 1e6

    native_resoln[filename] = resolution

with open('resolutions.pkl', 'wb') as resolution_file:
    pickle.dump(native_resoln, resolution_file)


print('Complete!')
