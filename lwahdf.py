import os
import sys
import h5py
import numpy
from datetime import datetime

from mnc.common import FS, CLOCK, NCHAN, CHAN_BW

__all__ = ['create_hdf5', 'set_frequencies', 'set_time',
           'set_polarization_products']


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
    f = h5py.File(filename, mode='w')
    
    # Top level attributes
    ## Observer and Project Info.
    f.attrs['ObserverID'] = 0
    f.attrs['ObserverName'] = ''
    f.attrs['ProjectID'] = ''
    f.attrs['SessionID'] = 0
    
    ## Station information
    f.attrs['StationName'] = 'ovro-lwa'
    
    ## File creation time
    f.attrs['FileCreation'] = datetime.utcnow().strftime("UTC %Y/%m/%d %H:%M:%S")
    f.attrs['FileGenerator'] = os.path.basename(__file__)
    
    ## Input file info.
    f.attrs['InputMetadata'] = ''
    
    # Observation group
    ## Create it if it doesn't exist
    obs = f.create_group('/Observation1')
    
    ## Target info.
    obs.attrs['TargetName'] = ''
    obs.attrs['RA'] = -99.0
    obs.attrs['RA_Units'] = 'hours'
    obs.attrs['Dec'] = -99.0
    obs.attrs['Dec_Units'] = 'degrees'
    obs.attrs['Epoch'] = 2000.0
    obs.attrs['Epoch'] = 2000.0
    obs.attrs['TrackingMode'] = 'Unknown'
    
    ## Observation info
    obs.attrs['ARX_Filter'] = -1.0
    obs.attrs['ARX_Gain1'] = -1.0
    obs.attrs['ARX_Gain2'] = -1.0
    obs.attrs['ARX_GainS'] = -1.0
    obs.attrs['Beam'] = beam
    obs.attrs['DRX_Gain'] = -1.0
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
    tun['freq'] = frequency.astype(numpy.float64)
    tun['freq'].attrs['Units'] = 'Hz'


def set_time(f, tint, count, format='unix', scale='utc'):
    """
    Set the integration time in seconds and create a data set to hold the time
    stamps.  Return the HDF5 data set.
    """
    
    obs = f.get('/Observation1', None)
    obs.attrs['tInt'] = tint
    obs.attrs['tInt_Units'] = 's'
    
    tim = obs.create_dataset('time', (count,), dtype=numpy.dtype({"names": ["int", "frac"],
                                                                  "formats": ["i8", "f8"]}))
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
        
        d = tun.create_dataset(p, (count, nchan), 'f4')
        d.attrs['axis0'] = 'time'
        d.attrs['axis1'] = 'frequency'
        data_products[i] = d
        data_products[p] = d
    return data_products
