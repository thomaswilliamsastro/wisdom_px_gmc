import os

import cmocean
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from vars import wisdom_dir, plot_dir, galaxy_dict, mask, vel_res, zoom

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir(os.path.join(wisdom_dir, 'new_reduction', 'derived'))

moments = [
    'mom0',
    'mom1',
    'ew'
]

fancy_names = {'mom0': r'Intensity (K km s$^{-1}$)',
               'mom1': r'Velocity (km s$^{-1}$)',
               'ew': r'Eff. width (km s$^{-1}$)'}

plot_name = os.path.join(plot_dir, 'data_overview')

fig, axes = plt.subplots(figsize=(3 * len(galaxy_dict.keys()), 3 * len(moments)),
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

        # Do a zoom if required
        if galaxy in zoom.keys():

            position = zoom[galaxy]['centre']
            if position is None:
                position = np.asarray(data.shape) / 2
            size = zoom[galaxy]['zoom']

            data = Cutout2D(data, position=position,
                            size=size)
            data = data.data

        # Colourmaps

        if moment in ['mom0', 'mom2', 'ew']:
            cmap = 'inferno'
            vmin, vmax = 0, np.nanpercentile(data, 99)  # np.nanpercentile(data, [1, 99])
        elif moment in ['mom1']:
            cmap = cmocean.cm.balance
            hdu.data -= np.nanmedian(data)
            vmin = np.nanpercentile(data, 1)
            vmax = -vmin
        else:
            raise Warning('Unknown cmap setup for %s' % moment)

        cbar_ticks = [int(vmin), int((vmin + vmax) / 2), int(vmax)]

        im = axes[j, i].imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

        if moment == moments[0]:
            axes[j, i].set_title(galaxy.upper())
        if galaxy == list(galaxy_dict.keys())[0]:
            axes[j, i].set_ylabel(fancy_names[moment])

        axes[j, i].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

        cbbox = inset_axes(axes[j, i],
                           width="70%", height="40%",
                           loc='lower center',
                           bbox_to_anchor=(0, -0.03, 1, 0.5),
                           bbox_transform=axes[j, i].transAxes,
                           )
        [cbbox.spines[k].set_visible(False) for k in cbbox.spines]
        cbbox.tick_params(axis='both', left=False, top=False,
                          right=False, bottom=False, labelleft=False,
                          labeltop=False, labelright=False, labelbottom=False)
        cbbox.set_facecolor([1, 1, 1, 0.8])
        cbaxes = inset_axes(cbbox, '70%', '20%', loc='upper center')

        # cbaxes = inset_axes(axes[j, i],
        #                     width="50%", height="5%",
        #                     loc='lower center',
        #                     bbox_to_anchor=(0, 0.075, 1, 0.5),
        #                     bbox_transform=axes[j, i].transAxes,
        #                     )
        cb = plt.colorbar(im, cax=cbaxes, orientation='horizontal',
                          ticks=cbar_ticks)

        # Add on scalebar
        if moment == 'mom0':

            phys_scale = 100

            # Add physical scalebar to top plots
            dist = galaxy_dict[galaxy]['info']['dist']
            pix_scale = np.abs(hdu.header['CDELT1'])
            phys_scalebar = np.degrees(phys_scale / (dist * 1e6)) / pix_scale

            scalebar = AnchoredSizeBar(axes[j, i].transData,
                                       phys_scalebar, '%d pc' % phys_scale, 'upper left',
                                       pad=0.25,
                                       borderpad=0.1,
                                       sep=4,
                                       # color='white',
                                       frameon=True,
                                       size_vertical=1,
                                       )

            axes[j, i].add_artist(scalebar)
        # plt.show()

# plt.tight_layout()

plt.subplots_adjust(hspace=0, wspace=0)

# plt.show()

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')

plt.close()

print('Complete!')
