import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from astropy.table import Table
from scipy.stats import gaussian_kde

from vars import wisdom_dir, plot_dir


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


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14
sns.set_color_codes()

os.chdir(wisdom_dir)

sun_tab = Table.read('sun_tableb1.dat',
                     format='ascii',
                     )

# Filter to our resolution, and remove things we can't weight
sun_tab = sun_tab[sun_tab['col2'] == 90]
sun_surf_den = 'col8'
sun_tab = sun_tab[sun_tab[sun_surf_den] > 0]

parameters = [
    'alpha_vir',
    'pressure'
]

plot_name = os.path.join(plot_dir, 'sun_comparison')

fig, axes = plt.subplots(1, 2,
                         figsize=(12, 4),
                         )

for i, parameter in enumerate(parameters):

    if parameter == 'alpha_vir':
        xlabel = r'$\alpha_\mathrm{vir}$'
        xlim = [10 ** -2.1, 10 ** 2.1]
        sun_col = 'col11'
    elif parameter == 'pressure':
        xlabel = r'$P_\mathregular{turb}$ (K cm$^{-3}$)'
        xlim = [10 ** 1.8, 10 ** 9.2]
        sun_col = 'col10'
    else:
        raise Warning('I dunno what a %s is' % parameter)

    # Calculate this for the Sun sample
    kde_range = np.arange(np.log10(xlim[0]), np.log10(xlim[1]), 0.01)
    kde = gaussian_kde(np.log10(sun_tab[sun_col]), weights=sun_tab[sun_surf_den], bw_method='silverman')
    kde_hist = kde.evaluate(kde_range)

    axes[i].plot(10 ** kde_range,
                 kde_hist,
                 c='k',
                 label='Sun+ 2020'
                 )

    gal_name, param, surf_dens = np.loadtxt('%s_90pc.txt' % parameter,
                                            unpack=True,
                                            dtype=str
                                            )

    param = np.array(param, dtype=float)
    surf_dens = np.array(surf_dens, dtype=float)

    print(parameter, weighted_quantile(param, quantiles=[0.5], sample_weight=surf_dens))

    galaxies = np.unique(gal_name)

    # colours = itertools.cycle(sns.color_palette('deep'))
    colours = iter(plt.cm.viridis(np.linspace(0, 1, len(galaxies))))

    for galaxy in galaxies:
        idx = np.where(gal_name == galaxy)

        c = next(colours)

        # Calculate the KDE per galaxy
        kde_range = np.arange(np.log10(xlim[0]), np.log10(xlim[1]), 0.01)
        kde = gaussian_kde(np.log10(param[idx]), weights=surf_dens[idx], bw_method='silverman')
        kde_hist = kde.evaluate(kde_range)

        axes[i].plot(10 ** kde_range,
                     kde_hist,
                     c=c,
                     label=galaxy.upper()
                     )

    axes[i].set_xscale('log')

    if parameter == 'alpha_vir':
        axes[i].set_xticks(ticks=[1e-1, 1, 10])

    if parameter == 'alpha_vir':
        axes[i].axvline(1, c='k', ls='--')
        axes[i].axvline(2, c='k', ls='-.')
    elif parameter == 'pressure':
        axes[i].axvline(2.9e4, c='k', ls='--')
        axes[i].axvline(5.1e6, c='k', ls='-.')

    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=3)
    axes[i].xaxis.set_major_locator(locmaj)

    locmin = matplotlib.ticker.LogLocator(base=10.0,
                                          subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                                          numticks=12)
    axes[i].xaxis.set_minor_locator(locmin)
    axes[i].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    axes[i].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    axes[i].set_xlim(xlim)

    axes[i].grid()

    axes[i].set_xlabel(xlabel)

    if i in [0, len(parameters) - 1]:
        axes[i].set_ylabel('Probability Density')
        if i == len(parameters) - 1:
            axes[i].yaxis.set_ticks_position('right')
            axes[i].yaxis.set_label_position('right')
    else:
        axes[i].yaxis.set_ticklabels([])

plt.legend(loc='center left',
           bbox_to_anchor=(1.2, 0.5),
           frameon=True,
           edgecolor='k',
           framealpha=1,
           fancybox=False,
           scatterpoints=1,
           numpoints=1,
           )

plt.tight_layout()
plt.subplots_adjust(wspace=0)

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.close()

# plt.show()

print('Complete!')
