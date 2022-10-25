# -*- coding: utf-8 -*-
"""
Pull K-band magnitudes from HYPERLEDA, and M_stars from z0MGS

@author: Tom Williams
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import wget
from astropy.io import ascii
from astroquery.irsa import Irsa

from vars import wisdom_dir, galaxies, plot_dir

os.chdir(wisdom_dir)

k_mags = {}
m_stars = {}

for galaxy in galaxies:

    # Query HYPERLEDA to pull out K-band mag

    if os.path.exists('fG.cgi'):
        os.remove('fG.cgi')
    hyperleda_query = wget.download('http://leda.univ-lyon1.fr/fG.cgi?n=a107&c=o&o=' + galaxy +
                                    '&a=htab')

    table = ascii.read('fG.cgi', format='html')

    k_mag = table[table['Band'] == 'Ks_2M']
    k_mag = float(k_mag['mag'][0].split()[0].split('(')[0])

    k_mags[galaxy] = k_mag

    # Query z0MGS for M_star

    query = Irsa.query_region(galaxy, catalog='z0mgsdr1index')
    if len(query) == 0:
        continue

    m_star = query['logmass'][0]

    m_stars[galaxy] = m_star

# Stellar masses from Jiayi's paper

phangs_mstar = np.array([2.1e10, 3e10, 0.76e10, 3.2e10, 3.6e10, 6.5e10, 7.4e10, 7.9e10, 3.9e10, 1.1e10, 8.1e10, 7.7e10,
                         16e10, 0.5e10])

m_star_list = np.array([m_stars[key] for key in m_stars.keys()])
plt.figure()

sns.kdeplot(m_star_list, color='r', label='This work')
sns.kdeplot(np.log10(phangs_mstar), color='b', label='Sun+ (18)')

plt.xlabel(r'$\log10(M_\ast [M_\odot])$')
plt.ylabel('Probability Density')

plt.legend(loc='upper left')

plot_name = os.path.join(plot_dir, 'mass_comparison')

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')

plt.close()

print('Complete!')
