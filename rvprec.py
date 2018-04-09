from numpy import *
import os
import glob
from collections import OrderedDict

def scale_flux(rad_m,eff,mag,exptime_s):
    """
    Scale the overall flux in a spectral chunk (from 0-mag photons/s/cm2/angstrom)
    :param rad_m: aperture radius in meters
    :param eff: efficiency (flat)
    :param mag: magnitude
    :param exptime_s: exposure time in seconds
    :return: scaling factor
    """
    rad_cm = rad_m * 100.
    rad_factor = (pi * rad_cm**2.) #/ (pi * 10.**2.)
    eff_factor = eff
    exptime_factor = exptime_s
    mag_factor = 10.**((0. - mag)/2.5)
    return rad_factor * eff_factor * exptime_factor * mag_factor

def find_nearest_idx(array,value):
    """
    Find the index of the value in an array that most closely matches a given value
    :param array: ndarray list of values
    :param value: value to match
    :return: index into array that yields closest value
    """
    idx = (abs(array-value)).argmin()
    return idx

def weighted_average(vals):
    """
    Compute the uncertainty of the weighted mean
    :param vals: ndarray of sigmas (for RV calculation)
    :return: uncertainty of weighted mean
    """
    arg1 = sum((1./vals)**2.)
    res = 1. / sqrt(arg1)
    return res

def do_we_have(teff,grav,qfiles,ffiles):
    """
    Figure out if there is are q and flux files with the requested Teff and gravity
    :param teff: effective temp of atmosphere model spectrum in K
    :param grav: log(g) of atmosphere model spectrum (cgs)
    :param qfiles: dict of q-factor files
    :param ffiles: dict of flux files
    :return: tuple (true/false,key for q-factor list, key for flux list)
    """
    for kk in qfiles.keys():
        if (qfiles[kk]['teff'] == teff) and (qfiles[kk]['grav'] == grav):
            for jj in ffiles.keys():
                if (qfiles[kk]['teff'] == teff) and (qfiles[kk]['grav'] == grav):
                    return(True,kk,jj)
    return(False,nan,nan)

def get_data(teff,grav,qfiles,ffiles):
    """
    Download data if it exists
    :param teff: effective temp of atmosphere model spectrum in K
    :param grav: log(g) of atmosphere model spectrum (cgs)
    :param qfiles: dict of q-factor files
    :param ffiles: dict of flux files
    :return: tuple (q data (dict), flux data (dict))
    """
    aa = do_we_have(teff,grav,qfiles,ffiles)
    if not aa[0]:
        raise ValueError('No file for this teff/grav')
    qdat = load(aa[1])[()]
    fdat = load(aa[2])[()]
    return(qdat,fdat)

def index_qs(directory='./output/'):
    files = glob.glob(os.path.join(directory,'*.npy'))
    oo1 = OrderedDict()
    for fi in files:
        tmpin = load(fi)
        tmp = tmpin[()]
        oo1[fi] = OrderedDict({'teff':tmp['teff'],'grav':tmp['grav']})
        del tmpin
    return oo1

def index_fs(directory='./output_fluxes/'):
    files = glob.glob(os.path.join(directory,'*.npy'))
    oo1 = OrderedDict()
    for fi in files:
        tmpin = load(fi)
        tmp = tmpin[()]
        oo1[fi] = OrderedDict({'teff':tmp['teff'],'grav':tmp['grav']})
        del tmpin
    return oo1


def calc_prec(qdat, fdat, wl1, wl2, resol, vsini_kms, rad_m, eff=0.1, mag=10, magtype='johnson,v',
              exptime_s=900, tell=0, mask_factor=0.1, sampling_pix_per_resel=5., beam_height_pix=5., rdn_pix=4.):
    """
    Calculate the shot+read noise limited achievable RV precision based on a set of pre-computed tables
    of quality factor (Bouchy+2001) and fluxes from BT-SETTL model spectra
    NOTES: this is a re-calculation Qs performed 4/2018.
    Haven't gotten around yet to scaling numbers of pixels excluded for telluric contamination
    (i.e. read noise will be higher than reality for these)

    :param qdat: q data (dict)
    :param fdat: flux data (dict)
    :param wl1: lower wavelength bound (angstroms), chunks of 200A: 3000-25000A
    :param wl2: upper wavelength bound (angstroms), chunks of 200A: 3000-25000A
    :param resol: resolution (2000,10000,30000,50000,55000,60000,80000,100000,110000,120000,0=native of model)
    :param vsini_kms: rotational velocity of spectrum in km/s (0,1,3,5,10,15,20)
    :param rad_m: aperture radius in m
    :param eff: full system efficiency
    :param mag: magnitude of source
    :param magtype: magnitude type (VRIJK) in form 'johnson,v', 'johnson,j', etc (derived from pysynphot)
    :param exptime_s: exposure time in seconds
    :param tell: telluric level to filter out (0.,0.9,0.95,0.99) e.g. anything where tellurics go below 0.9 is excluded
                from Q. 0=unfiltered. 30km/s on either side of an excluded pixel is also excluded (barycentric motion
                range approximation)
    :param mask_factor: unimplemented. will degrade Quality factor based on how much a given mask exploits
    :param sampling_pix_per_resel: sampling of resolution element in pixels (used to estimate total # pixels for read noise)
    :param beam_height_pix: height of beam on detector in pixels (estimate # of pixels for read noise)
    :param rdn_pix: read noise for a single pixel
    :return: dict with q, f, snr, rv precision for each chunk, as well as overall
    """

    # Wavelength limits for the spectral bins
    wls_q = array(qdat['chunk_lower_lims'])
    wls_q2 = array(qdat['chunk_upper_lims'])

    # Size of spectral bin (should be 200A)
    chunk_width = (wls_q2 - wls_q)[0]

    #check if wavelengths from Q and Flux arrays match
    wls_f = array(fdat['chunk_lower_lims'])
    if any(wls_q != wls_f):
        raise ValueError('diff wl arrs?')

    # Which resolutions, vsinis, and telluric levels do we have access to?
    resols = array(qdat['resols'])
    vsinis = array(qdat['vsinis'])
    tells = array(qdat['tell_levels'])

    # Which of each value shall we use?
    wl_i1 = find_nearest_idx(wls_q, wl1)
    wl_i2 = find_nearest_idx(wls_q, wl2) + 1
    resol_i = find_nearest_idx(resols, resol)
    vsini_i = find_nearest_idx(vsinis, vsini_kms)
    tell_i = find_nearest_idx(tells, tell)
    resol_use = resols[resol_i]
    vsini_use = vsinis[vsini_i]
    tell_use = tells[tell_i]
    wl1_use = wls_q[wl_i1]
    wl2_use = wls_q[wl_i2]
    scf = scale_flux(rad_m, eff, mag, exptime_s)

    print('Using \n wl_1: {} \n wl_2: {} \n Resol: {} \n vsini: {} \n tell: {} \n'.format(
        wl1_use, wl2_use, resol_use, vsini_use, tell_use))

    # calculate the quantities of interest for each spectral chunk
    qs_out = []
    fs_out = []
    sns_out = []
    sns_photon_out = []
    ws_out = []
    dv_out = []
    dv_photon_out = []
    sns_photon_out = []

    # loop over chunks of interest
    for wl_this in wls_q[wl_i1:wl_i2]:
        q1 = qdat['q_dict'][resol_use][vsini_use][tell_use][wl_this] # look up Q
        f1 = fdat['photlam_0'][magtype][wl_this] * scf # look up and scale flux

        # figure out number of pixels and read noise
        # how big is a resolution element in pixels?
        resel_fwhm = float(wl_this) / float(resol_use)
        # how many resolution elements in this chunk?
        n_resels = float(chunk_width) / float(resel_fwhm)
        # how many pixels (in dispersion) in this chunk?
        n_pix_horiz = n_resels * sampling_pix_per_resel
        # multiply by "height" of beam, assume rectangular beam
        n_pix_tot = n_pix_horiz * beam_height_pix

        # shot noise
        sn_photon = sqrt(f1)

        # shot + read noise
        sn_photon_rdn = f1 / sqrt(f1 + n_pix_tot * rdn_pix ** 2.)

        # calculate photon-limited precision (Bouchy+2001)
        dv_this_photon = 3e8 / (q1 * sn_photon)
        dv_this = 3e8 / (q1 * sn_photon_rdn) # photon + read noise

        # store the results for this chunk
        qs_out.append(q1)
        fs_out.append(f1)
        sns_out.append(sn_photon_rdn)
        sns_photon_out.append(sn_photon)
        ws_out.append(wl_this)
        dv_out.append(dv_this)
        dv_photon_out.append(dv_this_photon)

    # return all the info
    oo = {'qs': array(qs_out),
          'fs': array(fs_out),
          'sns': array(sns_out),
          'sns_photon': array(sns_photon_out),
          'ws': array(ws_out),
          'dv': array(dv_out),
          'dv_photon': array(dv_photon_out),
          'dv_all': weighted_average(array(dv_out)),
          'dv_all_photon': weighted_average(array(dv_photon_out))}
    return (oo)