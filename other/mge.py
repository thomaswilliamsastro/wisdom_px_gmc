from galpy.potential import Potential
import numpy as np
import matplotlib.pyplot as plt

G = 4.3e-6


class MGEPotential(Potential):
    """Class that implements a generic MGE potential

    Inputs:

        ml: Mass-to-light ratio (essentially the amplitude here).

        l: List of Gaussian amplitudes (in Lsun/pc^2)

        sigma: List of Gaussian widths (in pc)

        q: List of axial ratios (unitless!)

    """

    def __init__(self, l=1, sigma=1, q=1, ml=1,
                 normalize=False, ro=None, vo=None):

        Potential.__init__(self, amp=ml, ro=ro, vo=vo, amp_units='mass')
        self.l = l
        self.scale = self.l
        self.sigma = sigma
        self.q = q
        self.ml = ml
        if normalize or \
                (isinstance(normalize, (int, float)) \
                 and not isinstance(normalize, bool)):
            self.normalize(normalize)
        #         self.hasC= True
        #         self.hasC_dxdv= True
        self._nemo_accname = 'MGEPotential'

    def _evaluate(self, R, z, phi=0., t=0.):

        """
        NAME:
           _evaluate
        PURPOSE:
           evaluate the potential at R,z
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           Phi(R,z)

        """

        T = np.linspace(0, 1, 100)

        epsilon = 1 - self.q ** 2

        pot = 0

        for i in range(len(self.l)):
            integral = np.trapz(
                np.exp(-0.5 * T ** 2 / self.sigma[i] ** 2 * (R ** 2 + z ** 2 / (1 - epsilon[i] * T ** 2))) /
                np.sqrt(1 - epsilon[i] * T ** 2), T)

            pot += integral * self.l[i] / self.sigma[i]

        pot *= G * self.ml * np.sqrt(2 / np.pi)

        pot = -pot

        return pot

    def _Rforce(self, R, z, phi=0., t=0.):
        """
        NAME:
           _Rforce
        PURPOSE:
           evaluate the radial force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the radial force

        """

        T = np.linspace(0, 1, 100)

        epsilon = 1 - self.q ** 2

        r_force = 0

        for i in range(len(self.l)):
            integral = np.trapz(
                np.exp(-0.5 * T ** 2 / self.sigma[i] ** 2 * (R ** 2 + z ** 2 / (1 - epsilon[i] * T ** 2))) *
                (T ** 2 * R / self.sigma[i] ** 2) /
                np.sqrt(1 - epsilon[i] * T ** 2), T)

            r_force += integral * self.l[i] / self.sigma[i]

        r_force *= G * self.ml * np.sqrt(2 / np.pi)
        r_force = -r_force

        return r_force

    def _zforce(self, R, z, phi=0., t=0.):
        """
        NAME:
           _zforce
        PURPOSE:
           evaluate the vertical force for this potential
        INPUT:
           R - Galactocentric cylindrical radius
           z - vertical height
           phi - azimuth
           t - time
        OUTPUT:
           the vertical force
        """

        T = np.linspace(0, 1, 100)

        epsilon = 1 - self.q ** 2

        z_force = 0

        for i in range(len(self.l)):
            integral = np.trapz(
                np.exp(-0.5 * T ** 2 / self.sigma[i] ** 2 * (R ** 2 + z ** 2 / (1 - epsilon[i] * T ** 2))) *
                (T ** 2 * z / self.sigma[i] ** 2) / (1 - epsilon[i] * T ** 2) /
                np.sqrt(1 - epsilon[i] * T ** 2), T)

            z_force += integral * self.l[i] / self.sigma[i]

        z_force *= G * self.ml * np.sqrt(2 / np.pi)
        z_force = -z_force

        return z_force


dist = 68
ml = 2
inc = 60

sign = [1, 1, 1, 1, 1, 1, -1]
mge_l = 10 ** np.array([4.822, 4.073, 3.161, 3.533, 2.980, 2.308, 4.377])

mge_l = mge_l * sign

sigma = 10 ** np.array([0.345, 0.840, 1.269, 1.359, 1.728, 1.869, 0.348])
q = np.array([0.689, 0.599, 0.735, 0.109, 0.131, 0.360, 0.100])

arcsec_to_pc = dist * np.pi / 0.648

sigma *= arcsec_to_pc

# And into kpc for G

mge_l *= 1e6
sigma *= 1e-3

# Convert from Lsun/pc^2 to Lsun

mge_l *= 2 * np.pi * sigma ** 2 * q

pot = MGEPotential(l=mge_l, sigma=sigma, q=q, ml=ml, ro=1, vo=1, normalize=False)

from galpy.potential.plotRotcurve import calcRotcurve
from galpy.potential import vcirc

dists = np.linspace(0, 5, 100)
rotcurve = calcRotcurve(pot, dists)

plt.figure()

plt.plot(dists, rotcurve)

plt.show()
