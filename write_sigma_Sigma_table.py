import os

import numpy as np

from vars import galaxy_dict, wisdom_dir

os.chdir(wisdom_dir)

fit_dict = {}

resolutions = ['60pc', '90pc', '120pc']

for galaxy in galaxy_dict.keys():

    for resolution in resolutions:

        col_names = ['beta', 'A', 'scatter']
        for col_name in col_names:
            if resolution + '_' + col_name not in fit_dict.keys():
                fit_dict[resolution + '_' + col_name] = []

        file_name = os.path.join('fit_params', '%s_%s.txt' % (galaxy, resolution))

        if os.path.exists(file_name):

            beta, A, scatter = np.loadtxt(file_name, unpack=True)

            fit_dict[resolution + '_beta'].append(r'$%.3f\pm%.3f$' % (beta[0], (beta[1] + beta[2]) / 2))
            fit_dict[resolution + '_A'].append(r'$%.3f\pm%.3f$' % (A[0], (A[1] + A[2]) / 2))
            fit_dict[resolution + '_scatter'].append(r'$%.3f\pm%.3f$' % (scatter[0], (scatter[1] + scatter[2]) / 2))

        else:

            fit_dict[resolution + '_beta'].append('--')
            fit_dict[resolution + '_A'].append('--')
            fit_dict[resolution + '_scatter'].append('--')

# Now write this out

f = open('sigma_Sigma_fits.tex', 'w+')

f.write('\\begin{table}\n')
f.write('\\caption{Parameters for fits to the  $\\sigma/\\Sigma$ relationship for each galaxy (Eq. \\ref{eq:sun_sigma_sigma}). '
        'The subscript for each parameter indicates the resolution at which the fit was performed. '
        # 'In all cases, the statistical uncertainty in each parameter is negligible, so is omitted for brevity.'
        '\\label{tab:sigma_sigma_fits}}\n')
f.write('\\begin{tabular}{%s}\n' % ''.join('c' * (3 * len(resolutions) + 1)))
f.write('Galaxy')
for resolution in resolutions:
    f.write(' & ')
    f.write('$\\beta_{\\rm %s}$ & $A_{\\rm %s}$ & $\\Delta_{\\rm %s}$' % (resolution, resolution, resolution))
f.write('\\\\\n')
f.write('\\hline\n')

for i, galaxy in enumerate(galaxy_dict.keys()):

    f.write('%s' % galaxy.upper())

    for resolution in resolutions:
        f.write(' & ')
        f.write('%s & %s & %s' % (fit_dict[resolution + '_beta'][i],
                                  fit_dict[resolution + '_A'][i],
                                  fit_dict[resolution + '_scatter'][i]))

    f.write('\\\\\n')

f.write('\\hline\n')
f.write('\\end{tabular}\n')
f.write('\\end{table}\n')

print('Complete!')
