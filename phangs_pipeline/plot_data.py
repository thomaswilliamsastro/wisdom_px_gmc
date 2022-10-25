# -*- coding: utf-8 -*-
"""
Plot data to highlight the products we have

@author: Tom Williams
"""

import aplpy
import cmocean
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from astropy.io import fits

from vars import wisdom_dir, plot_dir

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

galaxy = 'ngc0383'
line_product = 'co10kms'
configs = ['12m+7m', '12m', '7m']
resolutions = ['', '60pc', '90pc', '120pc', '150pc']
products = ['mom0', 'mom1', 'mom2', 'ew']
masks = ['strict', 'broad']

radius = 4 / 3600

directory = os.path.join(wisdom_dir, 'new_reduction', 'derived', galaxy)
os.chdir(directory)

# First, show off the moments at native resolution, strict mask

plot_name = os.path.join(plot_dir, 'phangs_pipeline_moments')

fig = plt.figure(figsize=(12, 8))

for i, product in enumerate(products):

    file_name = '_'.join([galaxy, configs[0], line_product, masks[0], product]) + '.fits'
    err_file_name = file_name.replace(product, 'e' + product)

    hdu = fits.open(file_name)[0]

    if product != 'mom1':
        cmap = 'viridis'
        vmin, vmax = np.nanpercentile(hdu.data, [1, 99])
    else:
        cmap = cmocean.cm.balance
        median = np.nanmedian(hdu.data)
        vmax = np.nanpercentile(hdu.data, 86)
        vmin = 2 * median - vmax

    plot = aplpy.FITSFigure(file_name, figure=fig, subplot=(2, len(products), i + 1))
    plot.recenter(16.854, 32.4125, radius=radius)
    plot.show_colorscale(cmap=cmap, vmin=vmin, vmax=vmax)
    plot.axis_labels.hide()
    plot.ticks.hide()
    plot.tick_labels.hide()
    plot.set_title(product)

    plot = aplpy.FITSFigure(err_file_name, figure=fig, subplot=(2, len(products), i + 1 + len(products)))
    plot.recenter(16.854, 32.4125, radius=radius)
    plot.show_colorscale(cmap='viridis')
    plot.axis_labels.hide()
    plot.ticks.hide()
    plot.tick_labels.hide()
    plot.set_title(product + ' error')

plt.tight_layout()

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.close()

plot_name = os.path.join(plot_dir, 'phangs_pipeline_resolutions')

fig = plt.figure(figsize=(12, 4))

# Different resolutions

for i, resolution in enumerate(resolutions):

    file_name = '_'.join([galaxy, configs[0], line_product, resolution, masks[0], products[0]]) + '.fits'
    file_name = file_name.replace('__', '_')

    hdu = fits.open(file_name)[0]

    cmap = 'viridis'
    vmin, vmax = np.nanpercentile(hdu.data, [1, 99])

    plot = aplpy.FITSFigure(file_name, figure=fig, subplot=(1, len(resolutions), i + 1))
    plot.recenter(16.854, 32.4125, radius=radius)
    plot.show_colorscale(cmap=cmap, vmin=vmin, vmax=vmax)
    plot.axis_labels.hide()
    plot.ticks.hide()
    plot.tick_labels.hide()
    plot.set_title(resolution)

plt.tight_layout()
plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.close()

# Illustrate broad versus strict masking

plot_name = os.path.join(plot_dir, 'phangs_pipeline_masks')

fig = plt.figure(figsize=(6, 4))

for i, mask in enumerate(masks):

    file_name = '_'.join([galaxy, configs[1], line_product, mask, products[0]]) + '.fits'

    hdu = fits.open(file_name)[0]

    cmap = 'viridis'

    percentiles = [1, 99]
    if mask == 'broad':
        percentiles = [0.1, 99.9]
    vmin, vmax = np.nanpercentile(hdu.data, percentiles)

    plot = aplpy.FITSFigure(file_name, figure=fig, subplot=(1, len(masks), i + 1))
    plot.recenter(16.854, 32.4125, radius=radius)
    plot.show_colorscale(cmap=cmap, vmin=vmin, vmax=vmax)
    plot.axis_labels.hide()
    plot.ticks.hide()
    plot.tick_labels.hide()
    plot.set_title(mask)

plt.tight_layout()
plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.close()

# Different configurations

plot_name = os.path.join(plot_dir, 'phangs_pipeline_antenna_config')

fig = plt.figure(figsize=(6, 4))

for i, config in enumerate(configs):

    file_name = '_'.join([galaxy, config, line_product, masks[0], products[0]]) + '.fits'

    hdu = fits.open(file_name)[0]

    cmap = 'viridis'
    vmin, vmax = np.nanpercentile(hdu.data, [1, 99])

    plot = aplpy.FITSFigure(file_name, figure=fig, subplot=(1, len(configs), i + 1))
    plot.recenter(16.854, 32.4125, radius=radius)
    plot.show_colorscale(cmap=cmap, vmin=vmin, vmax=vmax)
    plot.axis_labels.hide()
    plot.ticks.hide()
    plot.tick_labels.hide()
    plot.set_title(config)

plt.tight_layout()
plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.close()

print('Complete!')
