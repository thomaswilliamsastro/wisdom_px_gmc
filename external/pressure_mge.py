import numpy as np
from astropy.io import fits
import astropy.units as u
import radio_beam
import scipy.stats as stats


def dist_ellipse(n, xc, yc, ratio, pa=0):  # original implementation (like DIST_ELLIPSE IDL function)

    """
    N = either  a scalar specifying the size of the N x N square output
              array, or a 2 element vector specifying the size of the
               M x N rectangular output array.
       XC,YC - Scalars giving the position of the ellipse center.   This does
               not necessarily have to be within the image
       RATIO - Scalar giving the ratio of the major to minor axis.   This
               should be greater than 1 for position angle to have its
               standard meaning.
    OPTIONAL INPUTS:
      POS_ANG - Position angle of the major axis in degrees, measured counter-clockwise
               from the Y axis.  For an image in standard orientation
               (North up, East left) this is the astronomical position angle.
               Default is 0 degrees.
    OUTPUT:
       IM - REAL*4 elliptical mask array, of size M x N.  THe value of each
               pixel is equal to the semi-major axis of the ellipse of center
                XC,YC, axial ratio RATIO, and position angle POS_ANG, which
               passes through the pixel.
    """

    ang = np.radians(pa + 90.)
    cosang = np.cos(ang)
    sinang = np.sin(ang)
    nx = n[1]
    ny = n[0]
    x = np.arange(-xc, nx - xc)
    y = np.arange(-yc, ny - yc)

    im = np.empty(n)
    xcosang = x * cosang
    xsinang = x * sinang

    for i in range(0, ny):
        xtemp = xcosang + y[i] * sinang
        ytemp = -xsinang + y[i] * cosang
        im[i, :] = np.sqrt((xtemp * ratio) ** 2 + ytemp ** 2)

    return im


class pressure_mge:
    def __init__(self, pa, inc, veldisp, distance):
        self.mom0 = None
        self.bmaj = None
        self.bmin = None
        self.bpa = None
        self.cellsize = 1
        self.freq = 230e9
        self.momclip = 0
        self.cent = [0, 0]
        self.rmax = 0
        self.surf = None
        self.sigma_arcsec = None
        self.qobs = None
        self.pa = pa
        self.inc = inc
        self.distance = distance
        self.ml = 1
        self.alpha_co = 4.35
        self.sdmap = None
        self.sdlim = 0
        self.stellar_voldens = None
        self.sd_profile = None
        self.veldisp = veldisp
        self.pressure = None
        self.pressure_err = None
        self.pressure_ul = None
        self.minpressure = None
        self.maxpressure = None
        self.meanpressure = None
        self.medianpressure = None
        self.meanpressure_areaweight = None
        self.totalmass = None
        self.sd_profile_err = None

    def load_fits(self, fitsfile, convert_to_sd=True, centre=None):
        if type(fitsfile) == str:
            hdulist = fits.open(fitsfile, ignore_blank=True)
        else:
            hdulist = fitsfile
        self.mom0 = hdulist[0].data.T
        hdr = hdulist[0].header
        try:
            beamtab = hdulist[1].data
            self.bmaj = np.median(beamtab['BMAJ'] / 3600.)
            self.bmin = np.median(beamtab['BMIN'] / 3600.)
            self.bpa = np.median(beamtab['BPA'])
        except:
            self.bmaj = hdr['BMAJ'] * 3600
            self.bmin = hdr['BMIN'] * 3600
            self.bpa = hdr['BPA']

        self.cellsize = hdr['CDELT2'] * 3600

        try:
            self.momclip = hdr['MOMCLIP']
        except KeyError:
            self.momclip = 0
        self.freq = hdr['RESTFRQ'] * u.Hz
        if not centre:
            self.cent = np.array(self.mom0.shape) / 2.
        else:
            self.cent = np.array(centre)
        self.rmax = (np.max(self.cent) * self.cellsize)

        if self.freq < 250e9 * u.Hz:
            self.alpha_co = 6.25  # *(u.s*u.pc*u.pc/u.K*u.km)
        else:
            self.alpha_co = 17.4  # *(u.s*u.pc*u.pc/u.K*u.km)

        if convert_to_sd:
            self.mom0_to_sd()
        else:
            self.sdmap = self.mom0 * np.cos(np.radians(self.inc))
            self.sdlim = self.momclip * np.cos(np.radians(self.inc))

    def load_mge(self, surf, sigma_arcsec, qobs, ml):
        self.surf = surf
        self.sigma_arcsec = sigma_arcsec
        self.qobs = qobs
        self.ml = ml

    def density_from_mge(self, rad_arc):
        pc_conversion_fac = 4.84 * self.distance  # pc/arcsec
        rad = rad_arc * pc_conversion_fac
        sigma = self.sigma_arcsec * pc_conversion_fac
        qintr2 = self.qobs ** 2 - np.cos(np.deg2rad(self.inc)) ** 2
        if np.any(qintr2 < 0.0):
            print('Inclination too low for deprojection')
        qintr = np.sqrt(qintr2) / np.sin(np.deg2rad(self.inc))
        if np.any(qintr <= 0.05): print('q < 0.05 components')
        surfcube = (self.surf * self.qobs) / (qintr * sigma * np.sqrt(2 * np.pi))
        dens = rad * 0.0
        for i in range(0, dens.size):
            dens[i] = ((surfcube * self.ml) * np.exp(((-1.) / (2 * (sigma ** 2))) * (rad[i] ** 2))).sum()
        self.stellar_voldens = dens

    def mom0_to_sd(self):
        beam = radio_beam.Beam(major=self.bmaj * u.arcsec, minor=self.bmin * u.arcsec, pa=self.bpa * u.deg)
        jy_beam_area = 1 * u.Jy / (beam.sr.to(u.arcsec * u.arcsec))
        jy_beam_conversion = (jy_beam_area).to(u.K, equivalencies=u.brightness_temperature(self.freq))
        self.sdmap = self.alpha_co * self.mom0 * jy_beam_conversion.value * np.cos(np.deg2rad(self.inc))
        self.sdlim = self.alpha_co * jy_beam_conversion.value * self.momclip * np.cos(np.deg2rad(self.inc))

    def gas_density_from_mom0(self, rad):
        distel = dist_ellipse(self.sdmap.shape, self.cent[0], self.cent[1], 1 / np.cos(np.deg2rad(self.inc)),
                              pa=self.pa) * self.cellsize
        dbeam = 0.5 * np.median(np.diff(rad))
        sdout = rad * 0.0
        sderror = sdout.copy()
        for i in range(0, rad.size):
            sdout[i] = np.nanmean(self.sdmap[(distel >= (rad[i] - dbeam)) & (distel < (rad[i] + dbeam))])
            sderror[i] = stats.sem(self.sdmap[(distel >= (rad[i] - dbeam)) & (distel < (rad[i] + dbeam))])

        self.sd_profile = sdout
        self.sd_profile_err = sderror

    def get_pressure(self):
        self.pressure = 273. * self.veldisp * np.clip(self.sd_profile, self.sdlim / 3., np.inf) * np.sqrt(
            self.stellar_voldens)
        self.pressure_err = np.sqrt(
            (self.sd_profile_err / self.sd_profile) ** 2 + (self.sdlim / self.sd_profile)) * self.pressure
        self.pressure_ul = np.array(self.sd_profile < self.sdlim / 3.)

    def get(self):
        if np.any(self.mom0) == None:
            raise Exception('Fits file not yet loaded')

        if np.any(self.qobs) == None:
            raise Exception('MGE not yet loaded')

        #### RADIUS ARRAY
        rad_arc = np.arange(0, self.rmax, self.bmaj) + self.bmaj / 2.

        #### GET GAS DENSITY w.r.t RADIUS
        self.gas_density_from_mom0(rad_arc)

        #### GET STELLAR DENSITY w.r.t RADIUS
        self.density_from_mge(rad_arc)

        #### GET PRESSURE w.r.t RADIUS
        self.get_pressure()

        self.minpressure = np.nanmin(self.pressure[~self.pressure_ul])
        self.maxpressure = np.nanmax(self.pressure[~self.pressure_ul])
        self.meanpressure = np.nanmean(self.pressure[~self.pressure_ul])
        self.medianpressure = np.nanmedian(self.pressure[~self.pressure_ul])

        radius_weight = ((rad_arc + self.bmaj / 2.) * 4.84 * self.distance) ** 2 - (
                    (rad_arc - self.bmaj / 2.) * 4.84 * self.distance) ** 2
        self.totalmass = (self.sd_profile * np.pi * radius_weight).sum()

        self.meanpressure_areaweight = np.average(self.pressure[~self.pressure_ul],
                                                  weights=radius_weight[~self.pressure_ul])

        return rad_arc, self.pressure, self.pressure_err, self.pressure_ul
