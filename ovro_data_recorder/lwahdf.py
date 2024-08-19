import os
import sys
import h5py
import numpy
from datetime import datetime
import logging

from astropy import time
from mnc.common import FS, CLOCK, NCHAN, CHAN_BW
from dsautils import dsa_store

__all__ = ['create_hdf5', 'set_frequencies', 'set_time',
           'set_polarization_products']


HDF5_CHUNK_SIZE_MB = 32


# Logging instance
lwahdf_logger = logging.getLogger('__main__')


# DsaStore instance
ls = dsa_store.DsaStore()


def create_hdf5(filename, beam, overwrite=False):
    """
    Create an empty HDF5 file with the right structure and groups.  Returns a
    h5py.File instance.
    """
    
    # Check for a pre-existing file
    if os.path.exists(filename):
        if not overwrite:
            raise RuntimeError("File '%s' already exists" % filename)
        else:
            os.unlink(filename)
            
    # Open the file
    f = h5py.File(filename, mode='w', libver='latest')

    # get keys with SDF contents. keys like "66_VOLT1"
    try:
        dd = ls.get_dict('/mon/observing/sdfdict')
    except Exception as e:
        dd = {}
        lwahdf_logger.warn(f"Could not access etcd values for /mon/observing/sdfdict: {str(e)}")

    sessionname = None
    session = {}
    observation = {}
    mjd_now = time.Time.now().mjd
    for kk, _ in dd.items():
        if dd[kk]['SESSION']['SESSION_DRX_BEAM'] == str(beam):
            for kkobs in dd[kk]['OBSERVATIONS'].keys():
                mjd_start = int(dd[kk]['OBSERVATIONS'][kkobs]['OBS_START_MJD'])+int(dd[kk]['OBSERVATIONS'][kkobs]['OBS_START_MPM'])/(1e3*24*3600)
                mjd_delta = int(dd[kk]['OBSERVATIONS'][kkobs]['OBS_DUR'])/(1e3*24*3600)
                mjd_stop = mjd_start + mjd_delta
                if mjd_now + mjd_delta/2 >= mjd_start and mjd_now + mjd_delta/2 < mjd_stop:  # align at midpoint of observation
                    sessionname = kk
                    session = dd[sessionname]['SESSION']
                    observation = dd[sessionname]['OBSERVATIONS'][kkobs]
                    break
            else:
                continue
            break

    logger.info(f"Session: {session}")
    logger.info(f"Observation: {observation}")

    # Top level attributes
    ## Observer and Project Info.
    f.attrs['ObserverID'] = int(session.get('PI_ID', 0))
    f.attrs['ObserverName'] = session.get('PI_NAME', '')
    f.attrs['ProjectID'] = session.get('PROJECT_ID', '')
    f.attrs['SessionID'] = int(session.get('SESSION_ID', 0))
    
    ## Station information
    f.attrs['StationName'] = 'ovro-lwa'
    
    ## File creation time
    f.attrs['FileCreation'] = datetime.utcnow().strftime("UTC %Y/%m/%d %H:%M:%S")
    f.attrs['FileGenerator'] = os.path.basename(__file__)
    
    ## Input file info.
    f.attrs['InputMetadata'] = ''
    f.attrs['ConfigFile'] = session.get('CONFIG_FILE', '')  # TODO: or put them above?
    f.attrs['CalDir'] = session.get('CAL_DIR', '')
    
    # Observation group
    ## Create it if it doesn't exist
    obs = f.create_group('/Observation1')
    
    ## Target info.
    obs.attrs['TargetName'] = observation.get('OBS_TARGET', '')
    obs.attrs['RA'] = float(observation.get('OBS_RA', -99.0))
    obs.attrs['RA_Units'] = 'hours'
    obs.attrs['Dec'] = float(observation.get('OBS_DEC', -99.0))
    obs.attrs['Dec_Units'] = 'degrees'
    obs.attrs['Epoch'] = 2000.0
    obs.attrs['TrackingMode'] = observation.get('OBS_MODE', 'Unknown')
    
    ## Observation info
    obs.attrs['ARX_Filter'] = -1.0
    obs.attrs['ARX_Gain1'] = -1.0
    obs.attrs['ARX_Gain2'] = -1.0
    obs.attrs['ARX_GainS'] = -1.0
    obs.attrs['Beam'] = beam
    obs.attrs['DRX_Gain'] = int(observation.get('OBS_DRX_GAIN', -1.0))
    obs.attrs['sampleRate'] = CLOCK
    obs.attrs['sampleRate_Units'] = 'Hz'
    obs.attrs['tInt'] = -1.0
    obs.attrs['tInt_Units'] = 's'
    obs.attrs['LFFT'] = NCHAN
    obs.attrs['nChan'] = 0
    obs.attrs['RBW'] = -1.0
    obs.attrs['RBW_Units'] = 'Hz'
    
    ## Create the "tuning"
    grp = obs.create_group('Tuning1')
    
    return f


def set_frequencies(f, frequency):
    """
    Define the frequency setup.
    """
    
    obs = f.get('/Observation1', None)
    obs.attrs['nChan'] = frequency.size
    obs.attrs['RBW'] = frequency[1] - frequency[0]
    obs.attrs['RBW_Units'] = 'Hz'
    
    tun = obs.get('Tuning1', None)
    tun['freq'] = frequency.astype('<f8')
    tun['freq'].attrs['Units'] = 'Hz'
    return tun['freq']


def set_time(f, tint, count, format='unix', scale='utc'):
    """
    Set the integration time in seconds and create a data set to hold the time
    stamps.  Return the HDF5 data set.
    """
    
    obs = f.get('/Observation1', None)
    obs.attrs['tInt'] = tint
    obs.attrs['tInt_Units'] = 's'
    
    tim = obs.create_dataset('time', (count,), dtype=numpy.dtype({"names": ["int", "frac"],
                                                                  "formats": ["<i8", "<f8"]}))
    tim.attrs['format'] = 'unix'
    tim.attrs['scale'] = 'utc'
    return tim


def set_polarization_products(f, pols, count):
    """
    Set the polarization products and create a data set for each.  Returns a
    dictionary of data sets keyed by the product name and its numeric index in
    the input.
    """
    
    obs = f.get('/Observation1', None)
    tun = obs.get('Tuning1', None)
    nchan = tun['freq'].size
    
    # Make sure we have a list
    if not isinstance(pols, (tuple, list)):
        pols = [p.strip().rstrip() for p in pols.split(',')]
        
    data_products = {}
    for i,p in enumerate(pols):
        p = p.replace('CR', 'XY_real')
        p = p.replace('CI', 'XY_imag')
        
        chunk_size = HDF5_CHUNK_SIZE_MB * 1024**2 // 4 // nchan
        chunk_size = max([1, chunk_size])
        chunk_size = min([count, chunk_size])
        
        d = tun.create_dataset(p, (count, nchan), '<f4', chunks=(chunk_size, nchan))
        d.attrs['axis0'] = 'time'
        d.attrs['axis1'] = 'frequency'
        data_products[i] = d
        data_products[p] = d
    return data_products
