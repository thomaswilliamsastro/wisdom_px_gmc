import os

from astropy.io import fits
import numpy as np

from vars import wisdom_dir, galaxy_dict

os.chdir(wisdom_dir)

masses = {
    'ngc0383': 11.82,
    'ngc0524': 11.40,
    'ngc1574': 10.79,
    'ngc3607': 11.34,
    'ngc4429': 11.17,
    'ngc4435': 10.69,
    'ngc4697': 11.07,
}

hubble_ts = {
    'ngc0383': -4.0,
    'ngc0524': -1.0,
    'ngc1574': -2.9,
    'ngc3607': -.32,
    'ngc4429': 0.0,
    'ngc4435': -2.1,
    'ngc4697': -5.0,
}

mge_refs = {
    'ngc0383': '\\cite{2019North}',
    'ngc0524': '\\cite{2019Smith}',
    'ngc1574': '\\cite{2023Ruffa}',
    'ngc3607': '\\cite{2013Scott}',
    'ngc4429': '\\cite{2018Davis}',
    'ngc4435': '\\cite{2013Scott}',
    'ngc4697': '\\cite{2017Davis}',
}

f = open('data_overview.tex', 'w+')

f.write('\\begin{table*}\n')
f.write('\\caption{Overview of data used in this study. \\label{tab:data_overview}}\n')
f.write('\\begin{tabular}{cccccccc}\n')
f.write('Galaxy & Distance & $T_{\\rm Hubble}$ (1) & $M_\\ast$ (2) & SFR (3) & $R_e$ (4) & Native resolution & MGE Ref. \\\\\n')
f.write(' & Mpc &  & $\\log_{10}(M_\\odot)$ & $\\log_{10}(M_\\odot~{\\rm yr}^{-1})$ & $^{\\prime \\prime}$ & pc & \\\\\n')
f.write('\\hline\n')

for key in galaxy_dict.keys():

    distance = galaxy_dict[key]['info']['dist']
    mass = masses[key]
    t = hubble_ts[key]
    mge_ref = mge_refs[key]
    r_eff = galaxy_dict[key]['info']['reff']
    sfr = galaxy_dict[key]['info']['sfr']

    hdu = fits.open(os.path.join('new_reduction',
                                 'derived',
                                 key,
                                 '%s_%s_%s_2p5kms.fits' % (key,
                                                           galaxy_dict[key]['antenna_config'],
                                                           galaxy_dict[key]['co_line'])),
                    )
    beam = hdu[0].header['BMAJ']

    res = np.radians(beam) * distance * 1e6

    hdu.close()

    f.write('%s & %.1f & %.1f & %.2f & %s & %s & %.2f & %s\\\\\n' % (key.upper(), distance, t, mass, sfr, r_eff, res, mge_ref))

f.write('\\hline\n')
f.write('\\end{tabular}\n')
f.write('\\\\(1) Hubble T morphology \\citep[HYPERLEDA,][]{2014Makarov}; '
        '(2) Total stellar mass: '
        'NGC0383 \\citep[MASSIVE,][]{2014Ma}, '
        'NGC0524, NGC1574 \\citep[z0MGS,][]{2019Leroy}, '
        'NGC3607, NGC4429, NGC4435, and NGC4697 \\citep[ATLAS$^{\\rm 3D}$,][]{2011Cappellari}, '
        '(3) Compiled by \\citet{2022Davis}, from \\citet{2014Davis, 2016Davis, 2019Leroy}, '
        '(4) 2MASS measured effective radius \\citep{2000Jarrett}.'
        '\n')
f.write('\\end{table*}\n')

f.close()

print('Complete!')

# \begin{table}
# \caption{Overview of data used in this study. {\bf Some missing} \label{tab:data_overview}}
# \begin{tabular}{ccccc}
# Galaxy & Distance & $T_{\rm Hubble}$ (1) & $M_\ast$ (2) & Native resolution \\
#  & Mpc &  & $\log_{10}(M_\odot)$ & pc \\
# \hline
# NGC0383 & 66.6 & -4.0 & -- & 44.42 \\
# NGC0524 & 23.3 & -1.0 & 10.99 & 38.11 \\
# NGC4429 & 16.5 & 0.0 & 10.8 & 12.75 \\
# NGC4697 & 11.4 & -5.0 & 10.76 & 29.72 \\
# \hline
# \end{tabular}
# \\(1) HYPERLEDA; (2) NGC0383 (MASSIVE), NGC3607, NGC4429, and NGC4435 (Atlas3D), NGC0524 (z0MGS)
# \end{table}
