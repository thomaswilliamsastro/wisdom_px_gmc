from __future__ import (
    division, print_function, absolute_import, unicode_literals)

from functools import partial
import numpy as np
from scipy.ndimage import generic_filter, binary_dilation, label
from scipy.stats import pearsonr
from scipy.special import erf
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import mad_std
from astropy.convolution import convolve_fft
from radio_beam import Beam
from spectral_cube import SpectralCube, Projection


def convolve_cube(
        incube, newbeam, mode='datacube',
        res_tol=0.0, min_coverage=0.8,
        append_raw=False, verbose=False, suppress_error=False):
    """
    Convolve a spectral cube or an rms noise cube to a specified beam.

    'datacube' mode: This is basically a wrapper around
    `~spectral_cube.SpectralCube.convolve_to()`,
    but it treats NaN values / edge effect in a more careful way
    (see the description of keyword 'min_coverage' below).

    'noisecube' mode: It handles rms noise cubes in a way that it
    correctly predicts the rms noise for the corresponding data cube
    convolved to the same specified beam.

    Parameters
    ----------
    incube : FITS HDU object or SpectralCube object
        Input spectral cube
    newbeam : radio_beam.Beam object
        Target beam to convolve to
    mode : {'datacube', 'noisecube'}, optional
        Whether the input cube is a data cube or an rms noise cube.
        In the former case, a direct convolution is performed.
        In the latter case, the convolution is done in a way that
        it returns the appropriate noise cube corresponding to
        the convolved data cube. (Default: 'datacube')
    res_tol : float, optional
        Tolerance on the difference between input/output resolution
        By default, a convolution is performed on the input cube
        when its native resolution is different from (sharper than)
        the target resolution. Use this keyword to specify a tolerance
        on resolution, within which no convolution will be performed.
        For example, res_tol=0.1 will allow a 10% tolerance.
    min_coverage : float or None, optional
        When the convolution meets NaN values or edges, the output is
        calculated by beam-weighted average. This keyword specifies
        a minimum beam covering fraction of valid (np.finite) pixels.
        Pixels with less beam covering fraction will be assigned NaNs.
        Default is an 80% beam covering fraction (min_coverage=0.8).
        If the user would rather use the interpolation strategy in
        `astropy.convolution.convolve_fft`, set this keyword to None.
        Note that the NaN pixels will be kept as NaN.
    append_raw : bool, optional
        Whether to append the raw convolved cube and weight cube.
        Default is not to append.
    verbose : bool, optional
        Whether to print the detailed processing log in terminal.
        Default is to not print.
    suppress_error : bool, optional
        Whether to suppress the error message when convolution is
        unsuccessful. Default is to not suppress.

    Returns
    -------
    outcube : FITS HDU objects or SpectralCube objects
        Convolved spectral cubes (when append_raw=False), or a 3-tuple
        including a masked verson, an unmaked version, and a coverage
        fraction cube (when append_raw=True).
        The output will be the same type of objects as the input.
    """

    if isinstance(incube, SpectralCube):
        cube = incube
    elif isinstance(incube, (fits.PrimaryHDU, fits.ImageHDU)):
        if 'BUNIT' in incube.header:
            unit = u.Unit(incube.header['BUNIT'], parse_strict='warn')
            if isinstance(unit, u.UnrecognizedUnit):
                unit = u.dimensionless_unscaled
        else:
            unit = u.dimensionless_unscaled
        cube = SpectralCube(
            data=incube.data*unit,
            wcs=WCS(incube.header), header=incube.header,
            allow_huge_operations=True).with_mask(
                np.isfinite(incube.data))
    else:
        raise ValueError(
            "`incube` needs to be either a SpectralCube object "
            "or a FITS HDU object")

    if mode not in ('datacube', 'noisecube'):
        raise ValueError("Invalid `mode` value: {}".format(mode))

    if (res_tol > 0) and (newbeam.major != newbeam.minor):
        raise ValueError(
            "Cannot handle a non-zero resolution torelance "
            "when the target beam is not round")

    if min_coverage is None:
        # Skip coverage check and preserve NaN values.
        # This uses the default interpolation strategy
        # implemented in 'astropy.convolution.convolve_fft'
        convolve_func = partial(
            convolve_fft, preserve_nan=True, allow_huge=True)
    else:
        # Do coverage check to determine the mask on the output
        convolve_func = partial(
            convolve_fft, nan_treatment='fill',
            boundary='fill', fill_value=0., allow_huge=True)

    tol = newbeam.major * np.array([1-res_tol, 1+res_tol])
    if ((tol[0] < cube.beam.major < tol[1]) and
        (tol[0] < cube.beam.minor < tol[1])):
        if verbose:
            print(
                "Native resolution within tolerance - "
                "copying original cube...")
        my_append_raw = False
        newcube = cube.unmasked_copy().with_mask(cube.mask.include())
    else:
        if verbose:
            print("Deconvolving beam...")
        try:
            beamdiff = newbeam.deconvolve(cube.beam)
        except ValueError as err:
            if suppress_error:
                if verbose:
                    print(
                        "{}\n"
                        "Old:  {:.3g} x {:.3g}  PA = {:.1f}\n"
                        "New:  {:.3g} x {:.3g}  PA = {:.1f}".format(
                            err,
                            cube.beam.major.to('arcsec'),
                            cube.beam.minor.to('arcsec'),
                            cube.beam.pa.to('deg'),
                            newbeam.major.to('arcsec'),
                            newbeam.minor.to('arcsec'),
                            newbeam.pa.to('deg')))
                    print("Exiting...")
                return
            else:
                raise ValueError(
                    "{}\n"
                    "Old:  {:.3g} x {:.3g}  PA = {:.1f}\n"
                    "New:  {:.3g} x {:.3g}  PA = {:.1f}".format(
                        err,
                        cube.beam.major.to('arcsec'),
                        cube.beam.minor.to('arcsec'),
                        cube.beam.pa.to('deg'),
                        newbeam.major.to('arcsec'),
                        newbeam.minor.to('arcsec'),
                        newbeam.pa.to('deg')))
        # ------------------------------------------------------------
        if verbose:
            print("Convolving cube...")
        if mode == 'datacube':
            # do convolution
            convcube = cube.convolve_to(
                newbeam, convolve=convolve_func)
            if min_coverage is not None:
                # divide the raw convolved image by the weight image
                # to correct for filling fraction
                my_append_raw = True
                wtcube = SpectralCube(
                    cube.mask.include().astype('float'),
                    cube.wcs, beam=cube.beam).with_mask(
                        np.ones(cube.shape).astype('?'))
                wtcube.allow_huge_operations = (
                    cube.allow_huge_operations)
                wtcube = wtcube.convolve_to(
                    newbeam, convolve=convolve_func)
                newcube = convcube / wtcube.unmasked_data[:]
                # mask all pixels w/ weight smaller than min_coverage
                threshold = min_coverage * u.dimensionless_unscaled
                newcube = newcube.with_mask(wtcube >= threshold)
            else:
                my_append_raw = False
                newcube = convcube
        elif mode == 'noisecube':
            # Analytically derive the rms noise cube for the low
            # resolution data cube. Steps are described as follows:
            # Step 1: square the high resolution noise cube
            cubesq = cube**2
            # Step 2: convolve the squared noise cube with a kernel
            #         that is sqrt(2) times narrower than the one
            #         used for data cube convolution
            beamdiff_small = Beam(
                major=beamdiff.major/np.sqrt(2),
                minor=beamdiff.minor/np.sqrt(2), pa=beamdiff.pa)
            newbeam_small = cube.beam.convolve(beamdiff_small)
            convcubesq = cubesq.convolve_to(
                newbeam_small, convolve=convolve_func)
            if min_coverage is not None:
                # divide the raw convolved image by the weight image
                # to correct for filling fraction
                my_append_raw = True
                wtcube_o = SpectralCube(
                    cube.mask.include().astype('float'),
                    cube.wcs, beam=cube.beam).with_mask(
                        np.ones(cube.shape).astype('?'))
                wtcube_o.allow_huge_operations = (
                    cube.allow_huge_operations)
                wtcube = wtcube_o.convolve_to(
                    newbeam_small, convolve=convolve_func)
                newcubesq = convcubesq / wtcube.unmasked_data[:]
                # mask all pixels w/ weight smaller than min_coverage
                # (here I force the masking of the noise cube to be
                #  consistent with that of the data cube)
                threshold = min_coverage * u.dimensionless_unscaled
                wtcube_d = wtcube_o.convolve_to(
                    newbeam, convolve=convolve_func)
                newcubesq = newcubesq.with_mask(wtcube_d >= threshold)
            else:
                my_append_raw = False
                newcubesq = convcubesq
            # Step 3: find the square root of the convolved noise cube
            convcube = np.sqrt(convcubesq)
            newcube = np.sqrt(newcubesq)
            # Step 4: apply a multiplicative factor, which accounts
            #         for the decrease in rms noise due to averaging
            convcube *= np.sqrt(cube.beam.sr/newbeam.sr).to('').value
            newcube *= np.sqrt(cube.beam.sr/newbeam.sr).to('').value

    if isinstance(incube, SpectralCube):
        if append_raw and my_append_raw:
            return newcube, convcube, wtcube
        else:
            return newcube
    elif isinstance(incube, (fits.PrimaryHDU, fits.ImageHDU)):
        if append_raw and my_append_raw:
            return newcube.hdu, convcube.hdu, wtcube.hdu
        else:
            return newcube.hdu


def calc_noise_in_cube(
        cube, masking_scheme='simple', mask=None,
        spatial_average_npix=None, spatial_average_nbeam=5.0,
        spectral_average_nchan=5, verbose=False):
    """
    Estimate rms noise in a (continuum-subtracted) spectral cube.

    Parameters
    ----------
    cube : SpectralCube object
        Input spectral cube (needs to be continuum-subtracted)
    masking_scheme : {'simple', 'user'}, optional
        Scheme for flagging signal in the cube. 'simple' means to flag
        all values above 3*rms or below -3*rms (default scheme);
        'user' means to use the user-specified mask (i.e., `mask`).
    mask : `np.ndarray` object, optional
        User-specified signal mask (ignored if masking_scheme!='user')
    spatial_average_npix : int, optional
        Size of the spatial averaging box, in terms of pixel number.
        If not None, `spatial_average_nbeam` will be ingored.
        (Default: None)
    spatial_average_nbeam : float, optional
        Size of the spatial averaging box, in the unit of beam FWHM
        (Default: 5.0)
    spectral_average_nchan : int, optional
        Size of the spectral averaging box, in terms of channel number
        (Default: 5)
    verbose : bool, optional
        Whether to print the detailed processing log in terminal.
        Default is to not print.

    Returns
    -------
    rmscube : SpectralCube object
        Spectral cube containing the rms noise at each ppv location
    """

    if masking_scheme not in ['simple', 'user']:
        raise ValueError(
            "'masking_scheme' should be specified as"
            "either 'simple' or 'user'")
    elif masking_scheme == 'user' and mask is None:
        raise ValueError(
            "'masking_scheme' set to 'user', yet "
            "no user-specified mask found")

    # extract negative values (only needed if masking_scheme='simple')
    if masking_scheme == 'simple':
        if verbose:
            print("Extracting negative values...")
        negmask = cube < (0 * cube.unit)
        negdata = cube.with_mask(negmask).filled_data[:].value
        negdata = np.stack([negdata, -1 * negdata], axis=-1)
    else:
        negdata = None

    # find rms noise as a function of channel
    if verbose:
        print("Estimating rms noise as a function of channel...")
    if masking_scheme == 'user':
        mask_v = mask
    elif masking_scheme == 'simple':
        rms_v = mad_std(negdata, axis=(1, 2, 3), ignore_nan=True)
        uplim_v = (3 * rms_v * cube.unit).reshape(-1, 1, 1)
        lolim_v = (-3 * rms_v * cube.unit).reshape(-1, 1, 1)
        mask_v = (
            ((cube - uplim_v) < (0 * cube.unit)) &
            ((cube - lolim_v) > (0 * cube.unit)))
    rms_v = cube.with_mask(mask_v).mad_std(axis=(1, 2)).quantity.value
    rms_v = generic_filter(
        rms_v, np.nanmedian, mode='constant', cval=np.nan,
        size=spectral_average_nchan)

    # find rms noise as a function of sightline
    if verbose:
        print("Estimating rms noise as a function of sightline...")
    if masking_scheme == 'user':
        mask_s = mask
    elif masking_scheme == 'simple':
        rms_s = mad_std(negdata, axis=(0, 3), ignore_nan=True)
        uplim_s = 3 * rms_s * cube.unit
        lolim_s = -3 * rms_s * cube.unit
        mask_s = (
            ((cube - uplim_s) < (0 * cube.unit)) &
            ((cube - lolim_s) > (0 * cube.unit)))
    rms_s = cube.with_mask(mask_s).mad_std(axis=0).quantity.value
    if spatial_average_npix is None:
        beamFWHM_pix = (
            cube.beam.major.to(u.deg).value /
            np.abs(cube.wcs.celestial.wcs.cdelt.min()))
        beamFWHM_pix = np.max([beamFWHM_pix, 3.])
        spatial_average_npix = int(
            spatial_average_nbeam * beamFWHM_pix)
    rms_s = generic_filter(
        rms_s, np.nanmedian, mode='constant', cval=np.nan,
        size=spatial_average_npix)

    # create rms noise cube from the tensor product of rms_v and rms_s
    if verbose:
        print("Creating rms noise cube (direct tensor product)...")
    rmscube = SpectralCube(
        np.einsum('i,jk', rms_v, rms_s), wcs=cube.wcs,
        header=cube.header.copy()).with_mask(cube.mask.include())
    rmscube.allow_huge_operations = cube.allow_huge_operations
    # adjust the normalization of the rms cube
    if masking_scheme == 'user':
        mask_n = mask
    elif masking_scheme == 'simple':
        rms_n = mad_std(negdata, ignore_nan=True)
        uplim_n = 3 * rms_n * cube.unit
        lolim_n = -3 * rms_n * cube.unit
        mask_n = (
            ((cube - uplim_n) < (0 * cube.unit)) &
            ((cube - lolim_n) > (0 * cube.unit)))
    rms_n = cube.with_mask(mask_n).mad_std().value
    rmscube /= rms_n

    # check unit
    if rmscube.unit != cube.unit:
        rmscube = rmscube * (cube.unit / rmscube.unit)

    return rmscube


def find_signal_in_cube(
        cube, noisecube, mask=None,
        nchan_hi=2, snr_hi=5, nchan_lo=2, snr_lo=2,
        prune_by_npix=None, prune_by_fracbeam=1.,
        expand_by_npix=None, expand_by_fracbeam=0.,
        expand_by_nchan=2, verbose=False):
    """
    Identify (positive) signal in a cube based on S/N ratio.

    This is a generalized version of the signal identification scheme
    described in Sun et al. (2018, ApJ, 860, 172).

    Here is a description of the scheme:
    1. Generate a 'core mask' by identifying detections with
       S/N >= `snr_hi` over at least `nchan_hi` consecutive channels.
       (if a user-specified mask exists, do this step inside the mask)
    2. Generate a 'wing mask' by identifying detections with
       S/N >= `snr_lo` over at least `nchan_lo` consecutive channels.
       (if a user-specified mask exists, do this step inside the mask)
    3. Dilate the 'core mask' inside the 'wing mask' to get
       a 'signal mask' that defines 'detections'.
    4. Label 'detections' by connectivity, and prune 'detections'
       in the 'signal mask' if projected area on sky smaller than
       a given number of pixels or a fraction of the beam area.
       (fraction given by `prune_by_npix` or `prune_by_fracbeam`)
    5. Expand the 'signal mask' along the spatial dimensions by
       a given number of pixels or a fraction of the beam FWHM.
       (fraction given by `expand_by_npix` or `expand_by_fracbeam`)
    6. Expand the 'signal mask' along the spectral dimension by
       a given number of channels.
       (# of channels given by `expand_by_nchan`)

    Parameters
    ----------
    cube : SpectralCube object
        Input spectral cube
    noisecube : SpectralCube object
        Estimated rms noise cube
    mask : `np.ndarray` object, optional
        User-specified mask, within which all steps are performed
    nchan_hi : int, optional
        # of consecutive channels specified for the 'core mask'
        (Default: 3)
    snr_hi : float, optional
        S/N threshold specified for the 'core mask'
        (Default: 3.5)
    nchan_lo : int, optional
        # of consecutive channels specified for the 'wing mask'
        (Default: 2)
    snr_lo : float, optional
        S/N threshold specified for the 'wing mask'
        (Default: 2.0)
    prune_by_npix : int, optional
        Threshold for pruning. All detections with projected area
        smaller than this number of pixels will be pruned.
        If not None, `prune_by_fracbeam` will be ignored.
        (Default: None)
    prune_by_fracbeam : float, optional
        Threshold for pruning. All detections with projected area
        smaller than this value times the beam area will be pruned.
        (Default: 1.0)
    expand_by_npix : int, optional
        Expand the signal mask along the spatial dimensions by this
        number of pixels. If not None, `expand_by_fracbeam` will be
        ignored. (Default: None)
    expand_by_fracbeam : float, optional
        Expand the signal mask along the spatial dimensions by this
        fraction times the beam HWHM
        (Default: 0.0)
    expand_by_nchan : int, optional
        Expand the signal mask along the spectral dimensions by this
        number of channels
        (Default: 2)
    verbose : bool, optional
        Whether to print the detailed processing log in terminal.
        Default is to not print.

    Returns
    -------
    outcube : SpectralCube object
        Input cube masked by the generated 'signal mask'.
    """

    if not cube.unit.is_equivalent(noisecube.unit):
        raise ValueError(
            "Incompatable units between 'cube' and 'noisecube'!")

    snr = cube.filled_data[:] / noisecube.filled_data[:]
    snr = snr.to(u.dimensionless_unscaled).value
    if mask is None:
        mask = np.ones_like(snr).astype('?')

    # generate core mask
    if verbose:
        print(
            "Generating core mask (S/N >= {} over {} consecutive "
            "channels)...".format(snr_hi, nchan_hi))
    mask_core = (snr > snr_hi)
    for iiter in range(nchan_hi-1):
        mask_core &= np.roll(mask_core, 1, 0)
    mask_core[:nchan_hi-1, :] = False
    for iiter in range(nchan_hi-1):
        mask_core |= np.roll(mask_core, -1, 0)
    mask_core &= mask

    # generate wing mask
    if verbose:
        print(
            "Generating wing mask (S/N >= {} over {} consecutive "
            "channels)...".format(snr_lo, nchan_lo))
    mask_wing = snr > snr_lo
    for iiter in range(nchan_lo-1):
        mask_wing &= np.roll(mask_wing, 1, 0)
    mask_wing[:nchan_lo-1, :] = False
    for iiter in range(nchan_lo-1):
        mask_wing |= np.roll(mask_wing, -1, 0)
    mask_wing &= mask

    # dilate core mask inside wing mask
    if verbose:
        print("Dilating core mask inside wing mask...")
    mask_signal = binary_dilation(
        mask_core, iterations=0, mask=mask_wing)

    # prune detections with small projected area on the sky
    if (prune_by_fracbeam > 0) or (prune_by_npix is not None):
        if verbose:
            print("Pruning detections with small projected area")
        if prune_by_npix is None:
            beamarea_pix = np.abs(
                cube.beam.sr.to(u.deg**2).value /
                cube.wcs.celestial.wcs.cdelt.prod())
            prune_by_npix = beamarea_pix * prune_by_fracbeam
        labels, count = label(mask_signal)
        for ind in np.arange(count)+1:
            if ((labels == ind).any(axis=0).sum() < prune_by_npix):
                mask_signal[labels == ind] = False

    # expand along spatial dimensions by a fraction of beam FWHM
    if (expand_by_fracbeam > 0) or (expand_by_npix is not None):
        if verbose:
            print("Expanding signal mask along spatial dimensions")
        if expand_by_npix is None:
            beamHWHM_pix = np.ceil(
                cube.beam.major.to(u.deg).value / 2 /
                np.abs(cube.wcs.celestial.wcs.cdelt.min()))
            expand_by_npix = int(beamHWHM_pix * expand_by_fracbeam)
        structure = np.zeros(
            [3, expand_by_npix*2+1, expand_by_npix*2+1])
        Y, X = np.ogrid[:expand_by_npix*2+1, :expand_by_npix*2+1]
        R = np.sqrt((X - expand_by_npix)**2 + (Y-expand_by_npix)**2)
        structure[1, :] = (R <= expand_by_npix)
        mask_signal = binary_dilation(
            mask_signal, iterations=1, structure=structure, mask=mask)

    # expand along spectral dimension by a number of channels
    if expand_by_nchan > 0:
        if verbose:
            print(
                "Expanding along spectral dimension by {} channels"
                "".format(expand_by_nchan))
        for iiter in range(expand_by_nchan):
            tempmask = np.roll(mask_signal, 1, axis=0)
            tempmask[0, :] = False
            mask_signal |= tempmask
            tempmask = np.roll(mask_signal, -1, axis=0)
            tempmask[-1, :] = False
            mask_signal |= tempmask
        mask_signal &= mask

    return cube.with_mask(mask_signal)


def calc_channel_corr(cube, mask=None, channel_lag=1):
    """
    Calculate channel-to-channel correlation coefficient (Pearson's r)

    Parameters
    ----------
    cube : SpectralCube object
        Input spectral cube
    mask : `np.ndarray` object, optional
        User-specified mask, within which to calculate r
    channel_lag : int
        Number of channel lag at which the correlation is calculated.
        Default is 1, which means to estimate correlation at one
        channel lag (i.e., between immediately adjacent channels)

    Returns
    -------
    r : float
        Pearson's correlation coefficient
    p-value : float
        Two-tailed p-value
    """
    if mask is None:
        mask = cube.mask.include()

    mask[-1, :] = False
    for i in np.arange(channel_lag):
        mask &= np.roll(mask, -1, axis=0)
    return pearsonr(
        cube.filled_data[mask],
        cube.filled_data[np.roll(mask, channel_lag, axis=0)])


def censoring_function(
    line_rms_width, channel_width, T_noise,
    spec_resp_kernel=[0, 1, 0],
    nchan_crit=2, snr_crit=2, scenario='optimistic'):
    """
    Calculate the shape of the censoring function.

    Parameters
    ----------
    line_rms_width: float or array-like
        The rms width of the Gaussian line profile
    channel_width: float or array-like
        The width of spectral channels, in the same unit as
        `line_rms_width`
    T_noise: float or array-like
        Noise temperature (1-sigma level)
    spec_resp_kernel: array-like with odd number of elements, optional
        A spectral response kernel designed to reproduce the
        appropriate channel-to-channel correlation
        (see Appendix A in Leroy et al. 2016, ApJ, 831, 16)
    nchan_crit: int, optional (default=2)
        # of channels used in determining significant detections
        (e.g., if the criterion is 2 consecutive channels above S/N=5,
        then `nchan_crit` should be set to 2)
    snr_crit: float, optional (default=2)
        S/R ratio used in determining significant detections
        (e.g., if the criterion is 2 consecutive channels above S/N=5,
        then `snr_crit` should be set to 5)
    scenario: {'optimistic', 'pessimistic'}, optional
        If 'optimistic' (default), this function will return the
        threshold below which the completeness is 0%.
        If 'pessimistic', this function will return the threshold
        above which the completeness is 100%.

    Returns
    -------
    line_intensity: float or array-like
        Completeness threshold in terms of integrated line intensity,
        carrying a unit of [T_noise] * [line_rms_width]
    """
    x = (np.asfarray(channel_width) /
         (2**0.5 * np.asfarray(line_rms_width)))

    if scenario == 'optimistic':
        # line center aligns perfectly with
        # the center of the `nchan_crit` consecutive channels
        uplim = nchan_crit / 2
        lolim = nchan_crit / 2 - 1
    elif scenario == 'pessimistic':
        # line center is off by half a channel width from
        # the center of the `nchan_crit` consecutive channels
        uplim = nchan_crit / 2 + 0.5
        lolim = nchan_crit / 2 - 0.5
    else:
        raise ValueError("Invalid `scenario`")

    kernel = np.asfarray(spec_resp_kernel).flatten()
    if (kernel.size % 2 == 0) or (kernel.sum() != 1):
        raise ValueError("Invalid spectral response kernel")

    ind = np.arange(-(kernel.size-1)/2, (kernel.size+1)/2)
    uplim = (uplim + ind) * x[..., np.newaxis]
    lolim = (lolim + ind) * x[..., np.newaxis]
    fraction = ((erf(uplim) - erf(lolim)) / 2 * kernel).sum(axis=-1)

    return np.asfarray(T_noise) * channel_width * snr_crit / fraction


def convolve_image(
        inimage, newbeam, res_tol=0.0, min_coverage=0.8,
        append_raw=False, verbose=False, suppress_error=False):
    """
    Convolve a 2D image to a specified beam.

    Very similar to `convolve_cube()`, but this function deals with
    2D images (i.e., projections) rather than 3D cubes.

    Parameters
    ----------
    inimage : FITS HDU object or ~spectral_cube.Projection object
        Input 2D image
    newbeam : radio_beam.Beam object
        Target beam to convolve to
    res_tol : float, optional
        Tolerance on the difference between input/output resolution
        By default, a convolution is performed on the input image
        when its native resolution is different from (sharper than)
        the target resolution. Use this keyword to specify a tolerance
        on resolution, within which no convolution will be performed.
        For example, res_tol=0.1 will allow a 10% tolerance.
    min_coverage : float or None, optional
        When the convolution meets NaN values or edges, the output is
        calculated by beam-weighted average. This keyword specifies
        a minimum beam covering fraction of valid (np.finite) values.
        Pixels with less beam covering fraction will be assigned NaNs.
        Default is an 80% beam covering fraction (min_coverage=0.8).
        If the user would rather use the interpolation strategy in
        `astropy.convolution.convolve_fft`, set this keyword to None.
        Note that the NaN pixels will be kept as NaN.
    append_raw : bool, optional
        Whether to append the raw convolved image and weight image.
        Default is not to append.
    verbose : bool, optional
        Whether to print the detailed processing log in terminal.
        Default is to not print.
    suppress_error : bool, optional
        Whether to suppress the error message when convolution is
        unsuccessful. Default is to not suppress.

    Returns
    -------
    outimage : FITS HDU objects or Projection objects
        Convolved 2D images (when append_raw=False), or a 3-tuple
        including a masked verson, an unmaked version, and a coverage
        fraction map (when append_raw=True).
        The output will be the same type of objects as the input.
    """

    if isinstance(inimage, Projection):
        proj = inimage
    elif isinstance(inimage, (fits.PrimaryHDU, fits.ImageHDU)):
        proj = Projection.from_hdu(inimage)
    else:
        raise ValueError(
            "`inimage` needs to be either a FITS HDU object "
            "or a spectral_cube.Projection object")

    if (res_tol > 0) and (newbeam.major != newbeam.minor):
        raise ValueError(
            "Cannot handle a non-zero resolution torelance "
            "when the target beam is not round")

    if min_coverage is None:
        # Skip coverage check and preserve NaN values.
        # This uses the default interpolation strategy
        # implemented in 'astropy.convolution.convolve_fft'
        convolve_func = partial(
            convolve_fft, preserve_nan=True, allow_huge=True)
    else:
        # Do coverage check to determine the mask on the output
        convolve_func = partial(
            convolve_fft, nan_treatment='fill',
            boundary='fill', fill_value=0., allow_huge=True)

    tol = newbeam.major * np.array([1-res_tol, 1+res_tol])
    if ((tol[0] < proj.beam.major < tol[1]) and
        (tol[0] < proj.beam.minor < tol[1])):
        if verbose:
            print(
                "Native resolution within tolerance - "
                "Copying original image...")
        my_append_raw = False
        newproj = proj.copy()
    else:
        if verbose:
            print("Convolving image...")
        try:
            convproj = proj.convolve_to(
                newbeam, convolve=convolve_func)
            if min_coverage is not None:
                # divide the raw convolved image by the weight image
                my_append_raw = True
                wtproj = Projection(
                    np.isfinite(proj.data).astype('float'),
                    wcs=proj.wcs, beam=proj.beam)
                wtproj = wtproj.convolve_to(
                    newbeam, convolve=convolve_func)
                newproj = convproj / wtproj.hdu.data
                # mask all pixels w/ weight smaller than min_coverage
                threshold = min_coverage * u.dimensionless_unscaled
                newproj[wtproj < threshold] = np.nan
            else:
                my_append_raw = False
                newproj = convproj
        except ValueError as err:
            if suppress_error:
                return
            else:
                raise ValueError(
                    "Unsuccessful convolution: {}\nOld: {}\nNew: {}"
                    "".format(err, proj.beam, newbeam))

    if isinstance(inimage, Projection):
        if append_raw and my_append_raw:
            return newproj, convproj, wtproj
        else:
            return newproj
    elif isinstance(inimage, (fits.PrimaryHDU, fits.ImageHDU)):
        if append_raw and my_append_raw:
            return newproj.hdu, convproj.hdu, wtproj.hdu
        else:
            return newproj.hdu