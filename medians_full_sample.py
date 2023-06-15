# -*- coding: utf-8 -*-
"""
KDE plot for the virial parameter nad pressure

@author: Tom Williams
"""

import os

import numpy as np

from vars import wisdom_dir

def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

os.chdir(wisdom_dir)

target_resolution = '60pc'

parameters = ['alpha_vir', 'pressure']

for parameter in parameters:
    gals, par_vals, surf_vals = np.loadtxt('%s_%s.txt' % (parameter, target_resolution),
                                           dtype=str, unpack=True)
    par_vals = par_vals.astype(float)
    surf_vals = surf_vals.astype(float)

    # Weight since we have uneven pixel numbers
    unique_gals = np.unique(gals)
    n_gal_weights = np.ones(len(gals))
    for gal in unique_gals:
        idx = np.where(gals == gal)[0]
        n_gal_weights[idx] = len(idx)

    quantiles = weighted_quantile(par_vals, quantiles=[0.16, 0.5, 0.84], sample_weight=surf_vals / n_gal_weights)
    print(quantiles[1], np.diff(quantiles))
    print('%.2e' % quantiles[1])
    print(['%.2e' % diff for diff in np.diff(quantiles)])

print('Complete!')
