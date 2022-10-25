# -*- coding: utf-8 -*-
"""
Plot velocity dispersion against H2 surface density

@author: Tom Williams
"""

import os

from astropy.table import Table
from astropy.io import ascii
import numpy as np

from vars import galaxies, wisdom_dir, resolutions

os.chdir(wisdom_dir)

fit_dict = {}

for galaxy in galaxies:
    for resolution in resolutions:

        resolution = str(resolution)

        col_names = ['beta', 'A', 'scatter']
        for col_name in col_names:
            if resolution + '_' + col_name not in fit_dict.keys():
                fit_dict[resolution + '_' + col_name] = []

        file_name = os.path.join('regrids', 'sigma_Sigma_fits', galaxy + '_' + resolution + '.txt')

        if os.path.exists(file_name):

            beta, A, scatter = np.loadtxt(file_name, unpack=True)

            # fit_dict[resolution + '_beta'].append(r'$%.3f\pm%.3f$' % (beta[0], beta[1]))
            # fit_dict[resolution + '_A'].append(r'$%.3f\pm%.3f$' % (A[0], A[1]))
            # fit_dict[resolution + '_scatter'].append(r'$%.3f\pm%.3f$' % (scatter[0], scatter[1]))

            fit_dict[resolution + '_beta'].append(r'$%.3f$' % beta[0])
            fit_dict[resolution + '_A'].append(r'$%.3f$' % A[0])
            fit_dict[resolution + '_scatter'].append(r'$%.3f$' % scatter[0])

        else:

            fit_dict[resolution + '_beta'].append('--')
            fit_dict[resolution + '_A'].append('--')
            fit_dict[resolution + '_scatter'].append('--')

# Now write this out

tab = Table()
tab.add_column(galaxies, name='Galaxy')

for key in fit_dict.keys():

    key_split = key.split('_')
    if key_split[0] != 'native':
        key_split[0] += r'\,pc'

    name = r'$'
    if key_split[1] == 'beta':
        name += r'\beta'
    elif key_split[1] == 'A':
        name += r'A'
    else:
        name += r'\Delta'
    name += r'_{\rm ' + key_split[0] + '}$'

    tab.add_column(fit_dict[key], name=name)

ascii.write(tab, 'sigma_Sigma_fits.tex', overwrite=True, format='latex',
            caption=r'Individual $\sigma/\Sigma$ fits for each galaxy. The subscript for each parameter indicates the '
                    r'resolution at which the fit was performed. The statistic error is negligible in all cases, so is'
                    r'omitted here for brevity. \label{tab:sigma_sigma_fits}',
            latexdict={'tabletype': 'table*',
                       'header_end': r'\hline',
                       'data_end': r'\hline',
                       },
            )