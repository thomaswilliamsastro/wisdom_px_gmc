# -*- coding: utf-8 -*-
"""
Create an overview table for the data

@author: Tom Williams
"""

import numpy as np
import os
import wget
from astropy.io import ascii
from astropy.table import Table
from astroquery.irsa import Irsa
import pickle

from vars import wisdom_dir, galaxies

os.chdir(wisdom_dir)

overview_table_file_name = 'overview_table'

if not os.path.exists(overview_table_file_name + '.fits'):

    overview_tab = Table()

    overview_tab.add_column(galaxies, name='galaxies')

    with open('distances.pkl', 'rb') as f:
        distances = pickle.load(f)

    with open('resolutions.pkl', 'rb') as f:
        resolutions = pickle.load(f)

    data_dict = {'galaxies': galaxies, 'distances': [], 'morph': [], 'mass': [],
                 'native_res': []}

    for galaxy in galaxies:

        # First, we want to pull in the HYPERLEDA morphology

        if os.path.exists('fG.cgi'):
            os.remove('fG.cgi')
        hyperleda_query = wget.download('http://leda.univ-lyon1.fr/fG.cgi?n=a102&c=o&o=' + galaxy +
                                        '&a=htab')

        table = ascii.read('fG.cgi', format='html')
        classification = float(table['type'][0].split('(')[0])

        data_dict['morph'].append(classification)

        # Query z0MGS for M_star

        query = Irsa.query_region(galaxy, catalog='z0mgsdr1index')
        if len(query) == 0:
            m_star = np.nan
        else:
            m_star = query['logmass'][0]

        data_dict['mass'].append(m_star)

        data_dict['distances'].append(distances[galaxy])
        data_dict['native_res'].append(resolutions[galaxy])

    overview_tab.add_columns([data_dict['distances'], data_dict['morph'], data_dict['mass'], data_dict['native_res']],
                             names=['dist', 'morph', 'mass', 'native_res'])

    overview_tab.write(overview_table_file_name + '.fits', overwrite=True)

else:

    overview_tab = Table.read(overview_table_file_name + '.fits')

overview_tab['native_res'] = ['%.2f' % res for res in overview_tab['native_res']]

ascii.write(overview_tab, 'data_overview.tex', overwrite=True, format='latex',
            names=['Galaxy', 'Distance', r'$T_{\rm Hubble}$ (1)', r'$M_\ast$ (2)', 'Native resolution'],
            caption=r'Overview of data used in this study. \label{tab:data_overview}',
            latexdict={'tabletype': 'table',
                       'units': {'Distance': 'Mpc',
                                 r'$M_\ast$ (2)': r'$\log_{10}(M_\odot)$',
                                 'Native resolution': 'pc'},
                       'header_end': r'\hline',
                       'data_end': r'\hline',
                       'tablefoot': r'\\(1) HYPERLEDA; (2) z0MGS'
                       },
            fill_values=('nan', '--')
            )
