# -*- coding: utf-8 -*-
"""
Make a nice plot of mom0 maps for talk

@author: Tom Williams
"""

import os

import numpy as np
from astropy.io import fits
import cmocean
import matplotlib.pyplot as plt
import matplotlib

from vars import wisdom_dir, galaxies, plot_dir

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(wisdom_dir)

n_cols = np.ceil(np.sqrt(len(galaxies)))

fig = plt.figure(figsize=(1.4 * n_cols * 2, n_cols * 2))

for i, galaxy in enumerate(galaxies):

    ax = plt.subplot(n_cols, n_cols, i + 1)

    hdu_file_name = os.path.join('regrids', galaxy + '_native_surf_dens.fits')
    hdu = fits.open(hdu_file_name)[0]
    data = hdu.data

    v_min, v_max = np.nanpercentile(data, 0.25), np.nanpercentile(data, 99.75)

    data[np.isnan(data)] = 0

    plt.imshow(data, origin='lower', cmap='inferno', vmin=v_min, vmax=v_max)
    plt.axis('off')

    plt.text(0.95, 0.95, galaxy,
             ha='right', va='top',
             color='white',
             transform=ax.transAxes)

plt.subplots_adjust(wspace=0, hspace=0)

# plt.show()

plot_name = os.path.join(plot_dir, 'data_overview')
plt.savefig(plot_name + '.png', bbox_inches='tight', transparent=True)
plt.savefig(plot_name + '.pdf', bbox_inches='tight')

plt.close()
print('Complete!')
