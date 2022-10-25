# -*- coding: utf-8 -*-
"""
Create a mock KinMS cube with known input parameters

@author: Tom Williams
"""

import os

from kinms import KinMS
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft
import matplotlib.pyplot as plt
from tqdm import tqdm

from vars import wisdom_dir

os.chdir(wisdom_dir)

# We'll simulate an exponential disc at a distance of 10Mpc, with a scale length of 100pc and a resolution of 30pc.
# So hopefully comparable to the WISDOM galaxies.

gal_dist = 10
arcsec_to_pc = gal_dist * 1e6 * np.pi / 180 / 3600

# Create a beam size

beam_size = 30 / arcsec_to_pc

radius = np.arange(0, 100, 0.1)

scale_radius = 100 / arcsec_to_pc

sb_prof = np.exp(-radius / scale_radius)

# Set up a velocity profile

# TODO: Push the turnover radius way out

vel = 210 * (2 / np.pi) * np.arctan(radius)

plt.figure()

plt.plot(radius, vel)

plt.show()

# Set up parameters to go into KinMS.

inc = 30
x_size, y_size = 64, 64
v_size = 800
dv = 2
pix_size = beam_size / 3
beam = [beam_size, beam_size, 0]
pos_ang = 45
gas_sigma = 10

# Make the cube

centre = [x_size / 2, y_size / 2, v_size / 2]

kin = KinMS(x_size, y_size, v_size, pix_size, dv, beamSize=beam, inc=inc, sbProf=sb_prof, gasSigma=gas_sigma,
            sbRad=radius, velProf=vel, posAng=pos_ang, verbose=True, fileName='data/KinMS')
model = kin.model_cube()

beam_std = beam_size / 2.355 / pix_size
kernel = Gaussian2DKernel(x_stddev=beam_std)

# Create a noise cube, and correlate with the beam
noise_cube = np.random.normal(loc=0, scale=0.1 * np.nanstd(model[~np.isnan(model)]), size=model.shape)

noise_cube_conv = np.zeros_like(noise_cube)
for i in tqdm(range(noise_cube.shape[-1])):
    noise_cube_conv[:, :, i] = convolve_fft(noise_cube[:, :, i], kernel)

model += noise_cube_conv
kin.save_fits(model, centre)
