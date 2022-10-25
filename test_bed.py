import os

from astropy.io import fits
import numpy as np
import astropy.units as u
from astroquery.ned import Ned
from astroquery.simbad import Simbad
import wget
from astropy.io import ascii
import glob

import matplotlib.pyplot as plt

from vars import wisdom_dir

os.chdir(wisdom_dir)

# Find the maximal WISDOM overlap with Atlas3D MGE models

atlas_mges = glob.glob('atlas_mges/*.txt')
atlas_mges = [s.split('_')[-1].split('.')[0] for s in atlas_mges]

# Read in the WISDOM galaxiess
wisdom_tab = ascii.read('WISDOM_basic_params.csv', data_start=1)
wisdom_galaxies = list(wisdom_tab['col1'])
atlas_wisdom_overlap = list(set(atlas_mges) & set(wisdom_galaxies))

# Add in WISDOM galaxiies
# atlas_wisdom_overlap.extend(['NGC0524', 'NGC0383', 'NGC1574])

print(sorted(atlas_wisdom_overlap))
# no

surf_hdu = fits.open('regrids/NGC3627_native_surf_dens.fits')[0]
eff_hdu = fits.open('regrids/NGC3627_native_eff_width.fits')[0]

beta_phangs, A_phangs, scat_phangs = 0.48, 0.66, 0.12
resolution_pc = 89 * u.pc

vir_x = np.linspace(np.nanmin(surf_hdu.data), np.nanmax(surf_hdu.data), 1000)

inv_b_to_alphavir = (5.77 * (80. / resolution_pc) * (u.Msun / u.pc ** 2 / (u.km / u.s) ** 2))

plt.figure()

plt.scatter(surf_hdu.data.flatten(),
            eff_hdu.data.flatten(),
            c='r', alpha=0.5)

plt.plot(vir_x, 10 ** (beta_phangs * np.log10((vir_x / 10 ** 2)) + A_phangs),
         c='k', ls='--',
         lw=2)
plt.plot(vir_x, (vir_x / inv_b_to_alphavir.value) ** 0.5,
         c='b', ls='--',
         lw=2)

plt.xscale('log')
plt.yscale('log')

plt.show()
