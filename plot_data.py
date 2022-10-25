import os
import glob

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import matplotlib
import cmocean

from vars import wisdom_dir, plot_dir, galaxy_dict, mask, vel_res

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(os.path.join(wisdom_dir, 'new_reduction', 'derived'))

moments = ['mom0', 'mom1', 'ew']

fancy_names = {'mom0': 'Intensity',
               'mom1': 'Velocity',
               'ew': 'Eff. width'}

plot_name = os.path.join(plot_dir, 'data_overview')

fig, axes = plt.subplots(figsize=(3*len(galaxy_dict.keys()), 3*len(moments)),
                         nrows=len(moments), ncols=len(galaxy_dict.keys()))

for i, galaxy in enumerate(galaxy_dict.keys()):

    for j, moment in enumerate(moments):

        antenna_config = galaxy_dict[galaxy]['antenna_config']
        co_line = galaxy_dict[galaxy]['co_line']

        file_name = os.path.join(galaxy, '%s_%s_%s_%s_%s_%s.fits' %
                                 (galaxy, antenna_config, co_line, vel_res, mask, moment))

        hdu = fits.open(file_name)[0]
        data = hdu.data

        # NaN out zeros for mom0

        if moment in ['mom0']:
            data[data == 0] = np.nan

        # Colourmaps

        if moment in ['mom0', 'mom2', 'ew']:
            cmap = 'inferno'
            vmin, vmax = np.nanpercentile(data, [1, 99])
        elif moment in ['mom1']:
            cmap = cmocean.cm.balance
            hdu.data -= np.nanmedian(data)
            vmin = np.nanpercentile(data, 1)
            vmax = -vmin
        else:
            raise Warning('Unknown cmap setup for %s' % moment)

        axes[j, i].imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

        if moment == moments[0]:
            axes[j, i].set_title(galaxy.upper())
        if galaxy == list(galaxy_dict.keys())[0]:
            axes[j, i].set_ylabel(fancy_names[moment])

        axes[j, i].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

plt.tight_layout()

plt.subplots_adjust(hspace=0)

# plt.show()

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')

plt.close()

print('Complete!')
