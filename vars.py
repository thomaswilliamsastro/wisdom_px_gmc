# -*- coding: utf-8 -*-
"""
Variables used frequently in the code

@author: Tom Williams
"""

import socket
import os
import glob
import numpy as np

hostname = socket.gethostname()

if hostname == 'mac-n-052':
    wisdom_dir = '/Users/williams/Documents/wisdom'
    plot_dir = '/Users/williams/Documents/wisdom/plots'
else:
    wisdom_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/wisdom'
    plot_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/wisdom/plots'

# Pull in useful names for galaxies

galaxy_dict = {}

os.chdir(os.path.join(wisdom_dir, 'new_reduction', 'derived'))

galaxies = glob.glob('*')
galaxies = sorted(galaxies)

# Preferred antenna configuration and CO lines
antenna_configs = ['12m+7m', '12m', '7m']
co_lines = ['co21', 'co32']

mask = 'strict'
vel_res = '2p5kms'

co_conv_factors = {'co21': 6.25,
                   'co32': 17.4}

galaxy_info = {'ngc0383': {'pa': 142.2, 'inc': 37.58, 'ml': 2.5,
                           'surf': [12913.045, 3996.8006, 5560.8414, 4962.2927, 2877.9230, 957.88407],
                           'sigma_arcsec': [0.0682436, 0.823495, 1.13042, 2.62962, 4.97714, 12.6523],
                           'qobs': [0.913337, 0.900000, 0.926674, 0.900000, 0.900000, 0.900000],
                           'r25': 16.23
                           },
               'ngc0404': {'pa': 37.2, 'inc': 20, 'ml': 0.66,
                           'surf': [303071.00, 258200.00, 7525.6001, 4395.6001, 2643.0000, 2632.1001, 16183.500,
                                    4477.1001, 3432.1001, 1722.1000, 1353.2100, 948.21002, 506.79401, 204.73599],
                           'sigma_arcsec': [0.0476858, 0.0501783, 0.0726512, 0.0836473, 0.120565, 0.190565, 0.250571,
                                            0.309951, 0.939995, 1.03987, 1.37274, 5.26472, 9.91885, 25.0187],
                           'qobs': [0.97399998, 0.97399998, 0.97399998, 0.97399998, 0.97399998, 0.97399998, 0.97399998,
                                    0.95800000, 0.95800000, 0.95800000, 0.95800000, 0.99699998, 0.99699998, 0.99699998],
                           'r25': 64
                           },
               'ngc0524': {'pa': 39.9, 'inc': 20.6, 'ml': 7,
                           'surf': [10 ** 4.336, 10 ** 3.807, 10 ** 4.431, 10 ** 3.914, 10 ** 3.638, 10 ** 3.530,
                                    10 ** 3.073,
                                    10 ** 2.450, 10 ** 1.832, 10 ** 1.300],
                           'sigma_arcsec': [10 ** -1.762, 10 ** -1.199, 10 ** -0.525, 10 ** -0.037, 10 ** 0.327,
                                            10 ** 0.629,
                                            10 ** 1.082, 10 ** 1.475, 10 ** 1.708, 10 ** 2.132],
                           'qobs': [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
                           'r25': 23.66
                           },
               'ngc3607': {'pa': 303.4, 'inc': 65, 'ml': 4.8,
                           'surf': [6040.9992, 4051.6947, 4738.1192, 2016.3855, 984.94731, 240.54797, 160.08261,
                                    80.807454, 29.180670],
                           'sigma_arcsec': [0.50912124, 1.4779759, 2.6574415, 5.0736632, 9.9389190, 18.378697,
                                            22.669179, 42.125922, 88.437680],
                           'qobs': [0.950000, 0.950000, 0.750000, 0.750000, 0.794637, 0.772915, 0.950000, 0.837034,
                                    0.950000],
                           'r25': 137,
                           },
               'ngc4429': {'pa': 93.2, 'inc': 66.8, 'ml': 6.59,
                           'surf': [21146.8, 1268.97, 4555.92, 2848.84, 1438.13, 658.099, 193.243, 26.5331, 6.95278],
                           'sigma_arcsec': [0.144716, 0.547251, 0.749636, 2.15040, 4.82936, 13.4457, 49.4016, 106.119,
                                            106.119],
                           'qobs': [0.675000, 0.950000, 0.400000, 0.700449, 0.780865, 0.615446, 0.400000, 0.400000,
                                    0.950000],
                           'r25': 48.84
                           },
               'ngc4435': {'pa': 194, 'inc': 70, 'ml': 3.9,
                           'surf': [5321.0533, 2728.0298, 3322.5759, 1822.5513, 109.99320, 618.08469, 187.81623,
                                    165.62275, 52.131347, 20.457966],
                           'sigma_arcsec': [0.51073228, 1.4723199, 2.9632470, 4.8266054, 7.3742823, 12.820829,
                                            18.608226, 20.388264, 46.932296, 46.932296],
                           'qobs': [0.918257, 0.950000, 0.750555, 0.781326, 0.950000, 0.400000, 0.810885,
                                    0.400000, 0.821091, 0.450046],
                           'r25': 90.6
                           },
               'ngc4697': {'pa': 246.2, 'inc': 76.1, 'ml': 2.14,
                           'surf': [10 ** 5.579, 10 ** 4.783, 10 ** 4.445, 10 ** 4.239, 10 ** 4.061, 10 ** 3.698,
                                    10 ** 3.314,
                                    10 ** 3.538],
                           'sigma_arcsec': [10 ** -1.281, 10 ** -0.729, 10 ** -0.341, 10 ** -0.006, 10 ** 0.367,
                                            10 ** 0.665,
                                            10 ** 0.809, 10 ** 1.191],
                           'qobs': [0.932, 1, 0.863, 0.726, 0.541, 0.683, 0.318, 0.589],
                           'r25': 39.51
                           },
               'ngc7052': {'pa': 64.3, 'inc': 74.8, 'ml': 3.88,
                           'surf': [10 ** 4.49, 10 ** 3.93, 10 ** 3.67, 10 ** 3.56],
                           'sigma_arcsec': [10 ** -1.76, 10 ** -0.23, 10 ** 0.14, 10 ** 0.6],
                           'qobs': [0.73, 0.77, 0.69, 0.71],
                           'r25': 20.25
                           }
               }

for galaxy in galaxies:

    galaxy_dict[galaxy] = {}

    file_found = False

    for antenna_config in antenna_configs:
        for co_line in co_lines:

            file_name = os.path.join(galaxy, '%s_%s_%s_%s_%s_%s.fits' %
                                     (galaxy, antenna_config, co_line, vel_res, mask, 'mom0'))

            if os.path.exists(file_name):
                file_found = True
                break

        if file_found:
            break

    galaxy_dict[galaxy]['antenna_config'] = antenna_config
    galaxy_dict[galaxy]['co_line'] = co_line
    galaxy_dict[galaxy]['info'] = galaxy_info[galaxy]

    # Finally, find resolutions

    file_names = glob.glob(os.path.join(galaxy, '*pc.fits'))
    resolutions = [file_name.split('_')[-1].split('.')[0] for file_name in file_names]
    resolutions = sorted(np.unique(resolutions))
    galaxy_dict[galaxy]['resolutions'] = resolutions

# galaxies = ['NGC4429',
#             'NGC0383',
#             'NGC4697',
#             'NGC4826',
#             'NGC5064',
#             'NGC0404',
#             'NGC0524',
#             'NGC4501',
#             'NGC1194',
#             'NGC1574',
#             'NGC5084',
#             'NGC7052',
#             'NGC3393',
#             'NGC0612',
#             'NGC0449',
#             'NGC3368',
#             ]
# galaxies = sorted(galaxies)
#
# # galaxies = ['KinMS_simcube']
# # galaxies = ['NGC3627']

resolutions = ['native', 45, 80, 120]

# Command to pull updates from astro-node4

# rsync -PvaSAD williams@astro-node4:/home/williams/wisdom /Users/williams/Documents
