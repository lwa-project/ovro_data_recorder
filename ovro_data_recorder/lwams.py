import os
import sys
import glob
import numpy
import shutil

from casacore.measures import measures
from casacore.tables import table, tableutil

from mnc.common import LWATime

__all__ = ['STOKES_CODES', 'NUMERIC_STOKES', 'get_zenith', 'get_zenith_uvw',
           'create_ms', 'update_time', 'update_pointing', 'update_data']


# Measurement set stokes name -> number
STOKES_CODES = {'I': 1,  'Q': 2,  'U': 3,  'V': 4, 
               'RR': 5, 'RL': 6, 'LR': 7, 'LL': 8,
               'XX': 9, 'XY':10, 'YX':11, 'YY':12}
               

# Measurement set stokes number -> name
NUMERIC_STOKES = { 1:'I',   2:'Q',   3:'U',   4:'V', 
                   5:'RR',  6:'RL',  7:'LR',  8:'LL',
                   9:'XX', 10:'XY', 11:'YX', 12:'YY'}


# Whether or not to flush the tables to disk before closing
FORCE_TABLE_FLUSH = False


def get_zenith(station, lwatime):
    """
    Given a Station instance and a LWATime instance, return the RA and Dec
    coordiantes of the zenith in radians (J2000).
    """
    
    # This should be compatible with what data2ms does/did.
    dm = measures()
    zenith = dm.direction('AZEL', '0deg', '90deg')
    position = dm.position(*station.casa_position)
    epoch = dm.epoch(*lwatime.casa_epoch)
    dm.doframe(zenith)
    dm.doframe(position)
    dm.doframe(epoch)
    pointing = dm.measure(zenith, 'J2000')
    return pointing['m0']['value'], pointing['m1']['value']


def get_zenith_uvw(station, lwatime):
    """
    Given a Station instance and a LWATime instance, return the (u,v,w)
    coordinates of the baselines.
    """
    
    # This should be compatible with what data2ms does/did.
    dm = measures()
    zenith = dm.direction('AZEL', '0deg', '90deg')
    position = dm.position(*station.casa_position)
    epoch = dm.epoch(*lwatime.casa_epoch)
    dm.doframe(zenith)
    dm.doframe(position)
    dm.doframe(epoch)
    
    nant = len(station.antennas)
    nbl = nant*(nant+1)//2
    pos = numpy.zeros((nant,3), dtype=numpy.float64)
    for i in range(nant):
        ant = station.antennas[i]
        position = dm.position('ITRF', *['%.6fm' % v for v in ant.ecef])
        baseline = dm.as_baseline(position)
        uvw = dm.to_uvw(baseline)
        pos[i,:] = uvw['xyz'].get_value()
        
    uvw = numpy.zeros((nbl,3), dtype=numpy.float64)
    k = 0
    for i in range(nant):
        a1 = pos[i,:]
        for j in range(i, nant):
            a2 = pos[j,:]
            uvw[k,:] = a1 - a2
            k += 1
            
    return uvw


class _MSConfig(object):
    """
    Class to wrap configuation information needed to fill in/update a measurement
    set.
    """
    
    def __init__(self, station, tint, freq, pols, nint=1):
        self.station = station
        self.tint = tint
        self.freq = freq
        self.pols = pols
        self.nint = nint
        
    @property
    def nant(self):
        """
        Antenna count
        """
        
        return len(self.station.antennas)
        
    @property
    def nbl(self):
        """
        Baseline count, including autocorrelations
        """
        
        return self.nant*(self.nant + 1) // 2
        
    @property
    def nchan(self):
        """
        Channel count
        """
        
        return self.freq.size
        
    @property
    def freq0(self):
        """
        First frequency channel
        """
        
        return self.freq[0]
        
    @property
    def chan_bw(self):
        """
        Channel bandwidth
        """
        
        return self.freq[1] - self.freq[0]
        
    @property
    def npol(self):
        """
        Polarization count
        """
        
        return len(self.pols)
    
    @property
    def settings(self):
        """
        ARX and F-engine settings used when recording the measurement set.
        """
        try:
            from observing import obsstate
            ss = obsstate.read_latest_setting()
            return ss['filename']
        except:
            return None


def create_ms(filename, station, tint, freq, pols, nint=1, overwrite=False):
    """
    Create an empty measurement set with the right structure and tables.
    """
    
    # Check for a pre-exiting file
    if os.path.exists(filename):
        if not overwrite:
            raise RuntimeError("File '%s' already exists" % filename)
        else:
            shutil.rmtree(filename)
            
    # Setup
    config = _MSConfig(station, tint, freq, pols, nint=nint)
    
    # Write some tables
    _write_main_table(filename, config)
    _write_antenna_table(filename, config)
    _write_polarization_table(filename, config)
    _write_observation_table(filename, config)
    _write_spectralwindow_table(filename, config)
    _write_misc_required_tables(filename, config)
    
    # Fixup the info and keywords for the main table
    tb = table(filename, readonly=False, ack=False)
    tb.putinfo({'type':'Measurement Set', 
               'readme':'This is a MeasurementSet Table holding measurements from a Telescope'})
    tb.putkeyword('MS_VERSION', numpy.float32(2.0))
    for tablename in sorted(glob.glob('%s/*' % filename)):
        if os.path.isdir(tablename):
            tname = os.path.basename(tablename)
            stb = table("%s/%s" % (filename, tname), ack=False)
            tb.putkeyword(tname, stb)
            stb.close()
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()


def update_time(filename, scan, start_time, centroid_time, stop_time):
    """
    Update the times inside a measurement set.
    """
    
    # Main table
    tb = table(filename, readonly=False, ack=False)
    nrow = tb.nrows()
    first_scan = tb.getcell('SCAN_NUMBER', 0)
    last_scan = tb.getcell('SCAN_NUMBER', nrow-1)
    nbl = nrow // (last_scan - first_scan + 1)
    tb.putcol('TIME', [start_time.measurementset,]*nbl, scan*nbl, nbl)
    tb.putcol('TIME_CENTROID', [centroid_time.measurementset,]*nbl, scan*nbl, nbl)
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
    # Feed table
    if scan == 0:
        tb = table(os.path.join(filename, "FEED"), readonly=False, ack=False)
        nant = tb.nrows()
        tb.putcol('TIME', [start_time.measurementset,]*nant, 0, nant)
        if FORCE_TABLE_FLUSH:
            tb.flush()
        tb.close()
        
    # Observation table
    tb = table(os.path.join(filename, "OBSERVATION"), readonly=False, ack=False)
    tb.putcell('TIME_RANGE', scan, [start_time.measurementset, stop_time.measurementset])
    tb.putcell('RELEASE_DATE', scan, start_time.measurementset)
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
    # Source table
    tb = table(os.path.join(filename, "SOURCE"), readonly=False, ack=False)
    tb.putcell('TIME', scan, start_time.measurementset)
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
    # Field table
    tb = table(os.path.join(filename, "FIELD"), readonly=False, ack=False)
    tb.putcell('TIME', scan, start_time.measurementset)
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()


def update_pointing(filename, scan, ra, dec):
    """
    Update the pointing for the first source in the measurement set to the
    provided RA and dec (in radians).
    """
    
    # Source table
    tb = table(os.path.join(filename, "SOURCE"), readonly=False, ack=False)
    tb.putcell('DIRECTION', scan, numpy.array([ra,dec]))
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
    # Field table
    tb = table(os.path.join(filename, "FIELD"), readonly=False, ack=False)
    for col in ('DELAY_DIR', 'PHASE_DIR', 'REFERENCE_DIR'):
        tb.putcell(col, scan, numpy.array([[ra,dec],]))
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()


def update_data(filename, scan, visibilities):
    """
    Update the visibilities in the main table.
    """
    
    # Main table
    tb = table(filename, readonly=False, ack=False)
    nbl = visibilities.shape[0]
    tb.putcol('DATA', visibilities, scan*nbl, nbl)
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()


def _write_main_table(filename, config):
    """
    Write the main data table.
    """
    
    station = config.station
    nant = config.nant
    nbl = config.nbl
    tint = config.tint
    freq = config.freq
    nchan = config.nchan
    pols = config.pols
    npol = config.npol
    nint = config.nint
    
    col1  = tableutil.makearrcoldesc('UVW', 0.0, 1, 
                                     comment='Vector with uvw coordinates (in meters)', 
                                     keywords={'QuantumUnits':['m','m','m'], 
                                               'MEASINFO':{'type':'uvw', 'Ref':'ITRF'}
                                               })
    col2  = tableutil.makearrcoldesc('FLAG', False, 2, 
                                     comment='The data flags, array of bools with same shape as data')
    col3  = tableutil.makearrcoldesc('FLAG_CATEGORY', False, 3,  
                                     comment='The flag category, NUM_CAT flags for each datum', 
                                     keywords={'CATEGORY':['',]})
    col4  = tableutil.makearrcoldesc('WEIGHT', 1.0, 1, 
                                     valuetype='float', 
                                     comment='Weight for each polarization spectrum')
    col5  = tableutil.makearrcoldesc('SIGMA', 9999., 1, 
                                     valuetype='float', 
                                     comment='Estimated rms noise for channel with unity bandpass response')
    col6  = tableutil.makescacoldesc('ANTENNA1', 0, 
                                     comment='ID of first antenna in interferometer')
    col7  = tableutil.makescacoldesc('ANTENNA2', 0, 
                                     comment='ID of second antenna in interferometer')
    col8  = tableutil.makescacoldesc('ARRAY_ID', 0, 
                                     comment='ID of array or subarray')
    col9  = tableutil.makescacoldesc('DATA_DESC_ID', 0, 
                                     comment='The data description table index')
    col10 = tableutil.makescacoldesc('EXPOSURE', 0.0, 
                                     comment='he effective integration time', 
                                     keywords={'QuantumUnits':['s',]})
    col11 = tableutil.makescacoldesc('FEED1', 0, 
                                     comment='The feed index for ANTENNA1')
    col12 = tableutil.makescacoldesc('FEED2', 0, 
                                     comment='The feed index for ANTENNA2')
    col13 = tableutil.makescacoldesc('FIELD_ID', 0, 
                                     comment='Unique id for this pointing')
    col14 = tableutil.makescacoldesc('FLAG_ROW', False, 
                                     comment='Row flag - flag all data in this row if True')
    col15 = tableutil.makescacoldesc('INTERVAL', 0.0, 
                                     comment='The sampling interval', 
                                     keywords={'QuantumUnits':['s',]})
    col16 = tableutil.makescacoldesc('OBSERVATION_ID', 0, 
                                     comment='ID for this observation, index in OBSERVATION table')
    col17 = tableutil.makescacoldesc('PROCESSOR_ID', -1, 
                                     comment='Id for backend processor, index in PROCESSOR table')
    col18 = tableutil.makescacoldesc('SCAN_NUMBER', 1, 
                                     comment='Sequential scan number from on-line system')
    col19 = tableutil.makescacoldesc('STATE_ID', -1, 
                                     comment='ID for this observing state')
    col20 = tableutil.makescacoldesc('TIME', 0.0, 
                                     comment='Modified Julian Day', 
                                     keywords={'QuantumUnits':['s',],
                                               'MEASINFO':{'type':'epoch', 'Ref':'UTC'}
                                               })
    col21 = tableutil.makescacoldesc('TIME_CENTROID', 0.0, 
                                     comment='Modified Julian Day', 
                                     keywords={'QuantumUnits':['s',],
                                               'MEASINFO':{'type':'epoch', 'Ref':'UTC'}
                                               })
    col22 = tableutil.makearrcoldesc("DATA", 0j, 2, 
                                     valuetype='complex',
                                     comment='The data column')
    
    desc = tableutil.maketabdesc([col1, col2, col3, col4, col5, col6, col7, col8, col9, 
                                  col10, col11, col12, col13, col14, col15, col16, 
                                  col17, col18, col19, col20, col21, col22])
    tb = table("%s" % filename, desc, nrow=nint*nbl, ack=False)
    
    fg = numpy.zeros((nint*nbl,npol,nchan), dtype=numpy.bool)
    fc = numpy.zeros((nint*nbl,npol,nchan,1), dtype=numpy.bool)
    uv = numpy.zeros((nint*nbl,3), dtype=numpy.float64)
    a1 = numpy.zeros((nint*nbl,), dtype=numpy.int32)
    a2 = numpy.zeros((nint*nbl,), dtype=numpy.int32)
    vs = numpy.zeros((nint*nbl,npol,nchan), dtype=numpy.complex64)
    wg = numpy.ones((nint*nbl,npol))
    sg = numpy.ones((nint*nbl,npol))*9999
    
    k = 0
    base_uvw = get_zenith_uvw(station, LWATime.now())   # Kind of an interesting choice
    for t in range(nint):
        uv[t*nbl:(t+1)*nbl,:] = base_uvw
        
        for i in range(nant):
            for j in range(i, nant):
                a1[k] = i
                a2[k] = j
                k += 1
                
    tb.putcol('UVW', uv, 0, nint*nbl)
    tb.putcol('FLAG', fg.transpose(0,2,1), 0, nint*nbl)
    tb.putcol('FLAG_CATEGORY', fc.transpose(0,3,2,1), 0, nint*nbl)
    tb.putcol('WEIGHT', wg, 0, nint*nbl)
    tb.putcol('SIGMA', sg, 0, nint*nbl)
    tb.putcol('ANTENNA1', a1, 0, nint*nbl)
    tb.putcol('ANTENNA2', a2, 0, nint*nbl)
    tb.putcol('ARRAY_ID', [0,]*nint*nbl, 0, nint*nbl)
    tb.putcol('DATA_DESC_ID', [0,]*nint*nbl, 0, nint*nbl)
    tb.putcol('EXPOSURE', [tint,]*nint*nbl, 0,nint* nbl)
    tb.putcol('FEED1', [0,]*nint*nbl, 0, nint*nbl)
    tb.putcol('FEED2', [0,]*nint*nbl, 0, nint*nbl)
    tb.putcol('FIELD_ID', [i for i in range(nint) for j in range(nbl)], 0, nint*nbl)
    tb.putcol('FLAG_ROW', [False,]*nint*nbl, 0, nint*nbl)
    tb.putcol('INTERVAL', [tint,]*nint*nbl, 0, nint*nbl)
    tb.putcol('OBSERVATION_ID', [i for i in range(nint) for j in range(nbl)], 0, nint*nbl)
    tb.putcol('PROCESSOR_ID', [-1,]*nint*nbl, 0, nint*nbl)
    tb.putcol('SCAN_NUMBER', [i+1 for i in range(nint) for j in range(nbl)], 0, nint*nbl)
    tb.putcol('STATE_ID', [-1,]*nint*nbl, 0, nint*nbl)
    tb.putcol('TIME', [0.0,]*nint*nbl, 0, nint*nbl)
    tb.putcol('TIME_CENTROID', [0.0,]*nint*nbl, 0, nint*nbl)
    tb.putcol('DATA', vs.transpose(0,2,1), 0, nint*nbl)
    
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
    # Data description
    
    col1 = tableutil.makescacoldesc('FLAG_ROW', False, 
                                    comment='Flag this row')
    col2 = tableutil.makescacoldesc('POLARIZATION_ID', 0, 
                                    comment='Pointer to polarization table')
    col3 = tableutil.makescacoldesc('SPECTRAL_WINDOW_ID', 0, 
                                    comment='Pointer to spectralwindow table')
    
    desc = tableutil.maketabdesc([col1, col2, col3])
    tb = table("%s/DATA_DESCRIPTION" % filename, desc, nrow=1, ack=False)
    
    tb.putcol('FLAG_ROW', [0,]*1, 0, 1)
    tb.putcol('POLARIZATION_ID', [0,]*1, 0, 1)
    tb.putcol('SPECTRAL_WINDOW_ID', [0,]*1, 0, 1)
    
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
def _write_antenna_table(filename, config):
    """
    Write the antenna table.
    """
    
    station = config.station
    nant = config.nant
    nbl = config.nbl
    tint = config.tint
    freq = config.freq
    nchan = config.nchan
    pols = config.pols
    npol = config.npol
    nint = config.nint
    
    col1 = tableutil.makearrcoldesc('OFFSET', 0.0, 1, 
                                    comment='Axes offset of mount to FEED REFERENCE point', 
                                    keywords={'QuantumUnits':['m','m','m'], 
                                              'MEASINFO':{'type':'position', 'Ref':'ITRF'}
                                    })
    col2 = tableutil.makearrcoldesc('POSITION', 0.0, 1,
                                    comment='Antenna X,Y,Z phase reference position', 
                                    keywords={'QuantumUnits':['m','m','m'], 
                                              'MEASINFO':{'type':'position', 'Ref':'ITRF'}
                                              })
    col3 = tableutil.makescacoldesc('TYPE', "ground-based", 
                                    comment='Antenna type (e.g. SPACE-BASED)')
    col4 = tableutil.makescacoldesc('DISH_DIAMETER', 2.0, 
                                    comment='Physical diameter of dish', 
                                    keywords={'QuantumUnits':['m',]})
    col5 = tableutil.makescacoldesc('FLAG_ROW', False, 
                                    comment='Flag for this row')
    col6 = tableutil.makescacoldesc('MOUNT', "alt-az", 
                                    comment='Mount type e.g. alt-az, equatorial, etc.')
    col7 = tableutil.makescacoldesc('NAME', "none", 
                                    comment='Antenna name, e.g. VLA22, CA03')
    col8 = tableutil.makescacoldesc('STATION', station.name, 
                                    comment='Station (antenna pad) name')
    
    desc = tableutil.maketabdesc([col1, col2, col3, col4, col5, col6, col7, col8])
    tb = table("%s/ANTENNA" % filename, desc, nrow=nant, ack=False)
    
    tb.putcol('OFFSET', numpy.zeros((nant,3)), 0, nant)
    tb.putcol('TYPE', ['GROUND-BASED,']*nant, 0, nant)
    tb.putcol('DISH_DIAMETER', [2.0,]*nant, 0, nant)
    tb.putcol('FLAG_ROW', [False,]*nant, 0, nant)
    tb.putcol('MOUNT', ['ALT-AZ',]*nant, 0, nant)
    tb.putcol('NAME', ['LWA%03i' % ant.id for ant in station.antennas], 0, nant)
    tb.putcol('STATION', [station.name,]*nant, 0, nant)
    
    for i,ant in enumerate(station.antennas):
        #tb.putcell('OFFSET', i, [0.0, 0.0, 0.0])
        tb.putcell('POSITION', i, ant.ecef)
        #tb.putcell('TYPE', i, 'GROUND-BASED')
        #tb.putcell('DISH_DIAMETER', i, 2.0)
        #tb.putcell('FLAG_ROW', i, False)
        #tb.putcell('MOUNT', i, 'ALT-AZ')
        #tb.putcell('NAME', i, ant.get_name())
        #tb.putcell('STATION', i, station.name)
        
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
def _write_polarization_table(filename, config):
    """
    Write the polarization table.
    """
    
    station = config.station
    nant = config.nant
    nbl = config.nbl
    tint = config.tint
    freq = config.freq
    nchan = config.nchan
    pols = config.pols
    npol = config.npol
    nint = config.nint
    
    # Polarization
    
    stks = numpy.array([STOKES_CODES[p] for p in pols])
    prds = numpy.zeros((2,npol), dtype=numpy.int32)
    for i in range(stks.size):
        stk = stks[i]
        if stk > 4:
            prds[0,i] = ((stk-1) % 4) / 2
            prds[1,i] = ((stk-1) % 4) % 2
        else:
            prds[0,i] = 1
            prds[1,i] = 1
            
    col1 = tableutil.makearrcoldesc('CORR_TYPE', 0, 1, 
                                    comment='The polarization type for each correlation product, as a Stokes enum.')
    col2 = tableutil.makearrcoldesc('CORR_PRODUCT', 0, 2, 
                                    comment='Indices describing receptors of feed going into correlation')
    col3 = tableutil.makescacoldesc('FLAG_ROW', False, 
                                    comment='flag')
    col4 = tableutil.makescacoldesc('NUM_CORR', npol, 
                                    comment='Number of correlation products')
    
    desc = tableutil.maketabdesc([col1, col2, col3, col4])
    tb = table("%s/POLARIZATION" % filename, desc, nrow=1, ack=False)
    
    tb.putcell('CORR_TYPE', 0, stks)
    tb.putcell('CORR_PRODUCT', 0, prds.T)
    tb.putcell('FLAG_ROW', 0, False)
    tb.putcell('NUM_CORR', 0, npol)
    
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
    # Feed
    
    col1  = tableutil.makearrcoldesc('POSITION', 0.0, 1, 
                                     comment='Position of feed relative to feed reference position', 
                                     keywords={'QuantumUnits':['m','m','m'], 
                                               'MEASINFO':{'type':'position', 'Ref':'ITRF'}
                                               })
    col2  = tableutil.makearrcoldesc('BEAM_OFFSET', 0.0, 2, 
                                     comment='Beam position offset (on sky but in antennareference frame)', 
                                     keywords={'QuantumUnits':['rad','rad'], 
                                               'MEASINFO':{'type':'direction', 'Ref':'J2000'}
                                               })
    col3  = tableutil.makearrcoldesc('POLARIZATION_TYPE', 'X', 1, 
                                     comment='Type of polarization to which a given RECEPTOR responds')
    col4  = tableutil.makearrcoldesc('POL_RESPONSE', 1j, 2,
                                     valuetype='complex',
                                     comment='D-matrix i.e. leakage between two receptors')
    col5  = tableutil.makearrcoldesc('RECEPTOR_ANGLE', 0.0, 1,  
                                     comment='The reference angle for polarization', 
                                     keywords={'QuantumUnits':['rad',]})
    col6  = tableutil.makescacoldesc('ANTENNA_ID', 0, 
                                     comment='ID of antenna in this array')
    col7  = tableutil.makescacoldesc('BEAM_ID', -1, 
                                     comment='Id for BEAM model')
    col8  = tableutil.makescacoldesc('FEED_ID', 0, 
                                     comment='Feed id')
    col9  = tableutil.makescacoldesc('INTERVAL', 0.0, 
                                     comment='Interval for which this set of parameters is accurate', 
                                     keywords={'QuantumUnits':['s',]})
    col10 = tableutil.makescacoldesc('NUM_RECEPTORS', 2, 
                                     comment='Number of receptors on this feed (probably 1 or 2)')
    col11 = tableutil.makescacoldesc('SPECTRAL_WINDOW_ID', -1, 
                                     comment='ID for this spectral window setup')
    col12 = tableutil.makescacoldesc('TIME', 0.0, 
                                     comment='Midpoint of time for which this set of parameters is accurate', 
                                     keywords={'QuantumUnits':['s',], 
                                               'MEASINFO':{'type':'epoch', 'Ref':'UTC'}
                                               })
    
    desc = tableutil.maketabdesc([col1, col2, col3, col4, col5, col6, col7, col8, 
                                  col9, col10, col11, col12])
    tb = table("%s/FEED" % filename, desc, nrow=nant, ack=False)
    
    presp = numpy.zeros((nant,2,2), dtype=numpy.complex64)
    if stks[0] > 8:
        ptype = [['X', 'Y'] for i in range(nant)]
        presp[:,0,0] = 1.0
        presp[:,0,1] = 0.0
        presp[:,1,0] = 0.0
        presp[:,1,1] = 1.0
    elif stks[0] > 4:
        ptype = [['R', 'L'] for i in range(nant)]
        presp[:,0,0] = 1.0
        presp[:,0,1] = -1.0j
        presp[:,1,0] = 1.0j
        presp[:,1,1] = 1.0
    else:
        ptype = [['X', 'Y'] for i in range(nant)]
        presp[:,0,0] = 1.0
        presp[:,0,1] = 0.0
        presp[:,1,0] = 0.0
        presp[:,1,1] = 1.0
        
    tb.putcol('POSITION', numpy.zeros((nant,3)), 0, nant)
    tb.putcol('BEAM_OFFSET', numpy.zeros((nant,2,2)), 0, nant)
    tb.putcol('POLARIZATION_TYPE', numpy.array(ptype, dtype='S'), 0, nant)
    tb.putcol('POL_RESPONSE', presp, 0, nant)
    tb.putcol('RECEPTOR_ANGLE', numpy.zeros((nant,2)), 0, nant)
    tb.putcol('ANTENNA_ID', list(range(nant)), 0, nant)
    tb.putcol('BEAM_ID', [-1,]*nant, 0, nant)
    tb.putcol('FEED_ID', [0,]*nant, 0, nant)
    tb.putcol('INTERVAL', [tint,]*nant, 0, nant)
    tb.putcol('NUM_RECEPTORS', [2,]*nant, 0, nant)
    tb.putcol('SPECTRAL_WINDOW_ID', [-1,]*nant, 0, nant)
    tb.putcol('TIME', [0.0,]*nant, 0, nant)
    
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
def _write_observation_table(filename, config):
    """
    Write the observation table.
    """
    
    station = config.station
    nant = config.nant
    nbl = config.nbl
    tint = config.tint
    freq = config.freq
    nchan = config.nchan
    pols = config.pols
    npol = config.npol
    nint = config.nint
    
    # Observation
    
    col1 = tableutil.makearrcoldesc('TIME_RANGE', 0.0, 1, 
                                    comment='Start and end of observation', 
                                    keywords={'QuantumUnits':['s',], 
                                              'MEASINFO':{'type':'epoch', 'Ref':'UTC'}
                                              })
    col2 = tableutil.makearrcoldesc('LOG', 'none', 1,
                                    comment='Observing log')
    col3 = tableutil.makearrcoldesc('SCHEDULE', 'none', 1,
                                    comment='Observing schedule')
    col4 = tableutil.makescacoldesc('FLAG_ROW', False, 
                                    comment='Row flag')
    col5 = tableutil.makescacoldesc('OBSERVER', station.name, 
                                    comment='Name of observer(s)')
    col6 = tableutil.makescacoldesc('PROJECT', station.name, 
                                    comment='Project identification string')
    col7 = tableutil.makescacoldesc('RELEASE_DATE', 0.0, 
                                    comment='Release date when data becomes public', 
                                    keywords={'QuantumUnits':['s',], 
                                              'MEASINFO':{'type':'epoch', 'Ref':'UTC'}
                                              })
    col8 = tableutil.makescacoldesc('SCHEDULE_TYPE', 'all-sky', 
                                    comment='Observing schedule type')
    col9 = tableutil.makescacoldesc('TELESCOPE_NAME', station.name, 
                                    comment='Telescope Name (e.g. WSRT, VLBA)')
    
    desc = tableutil.maketabdesc([col1, col2, col3, col4, col5, col6, col7, col8, col9])
    tb = table("%s/OBSERVATION" % filename, desc, nrow=nint, ack=False)
    
    tb.putcol('TIME_RANGE', numpy.zeros((nint,2)), 0, nint)
    tb.putcol('LOG', numpy.array([['Not provided',],]*nint, dtype='S'), 0, nint)
    tb.putcol('SCHEDULE', numpy.array([['Not provided',],]*nint, dtype='S'), 0, nint)
    tb.putcol('FLAG_ROW', [False,]*nint, 0, nint)
    tb.putcol('OBSERVER', [os.path.basename(__file__),]*nint, 0, nint)
    tb.putcol('PROJECT', [station.name+' all-sky',]*nint, 0, nint)
    tb.putcol('RELEASE_DATE', [0.0,]*nint, 0, nint)
    tb.putcol('SCHEDULE_TYPE', ['all-sky',]*nint, 0, nint)
    tb.putcol('TELESCOPE_NAME', [station.name,]*nint, 0, nint)
    
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
    # Source
    
    col1  = tableutil.makearrcoldesc('DIRECTION', 0.0, 1,
                                     comment='Direction (e.g. RA, DEC).', 
                                     keywords={'QuantumUnits':['rad','rad'], 
                                               'MEASINFO':{'type':'direction', 'Ref':'J2000'}
                                               })
    col2  = tableutil.makearrcoldesc('PROPER_MOTION', 0.0, 1,
                                     comment='Proper motion', 
                                     keywords={'QuantumUnits':['rad/s',]})
    col3  = tableutil.makescacoldesc('CALIBRATION_GROUP', 0, 
                                     comment='Number of grouping for calibration purpose.')
    col4  = tableutil.makescacoldesc('CODE', "none", 
                                     comment='Special characteristics of source, e.g. Bandpass calibrator')
    col5  = tableutil.makescacoldesc('INTERVAL', 0.0, 
                                     comment='Interval of time for which this set of parameters is accurate', 
                                     keywords={'QuantumUnits':['s',]})
    col6  = tableutil.makescacoldesc('NAME', "none", 
                                     comment='Name of source as given during observations')
    col7  = tableutil.makescacoldesc('NUM_LINES', 0, 
                                     comment='Number of spectral lines')
    col8  = tableutil.makescacoldesc('SOURCE_ID', 0, 
                                     comment='Source id')
    col9  = tableutil.makescacoldesc('SPECTRAL_WINDOW_ID', -1, 
                                     comment='ID for this spectral window setup')
    col10 = tableutil.makescacoldesc('TIME', 0.0,
                                     comment='Midpoint of time for which this set of parameters is accurate.', 
                                     keywords={'QuantumUnits':['s',], 
                                               'MEASINFO':{'type':'epoch', 'Ref':'UTC'}
                                               })
    col11 = tableutil.makearrcoldesc('TRANSITION', 'none', 1, 
                                     comment='Line Transition name')
    col12 = tableutil.makearrcoldesc('REST_FREQUENCY', 1.0, 1, 
                                     comment='Line rest frequency', 
                                     keywords={'QuantumUnits':['Hz',], 
                                               'MEASINFO':{'type':'frequency', 
                                                           'Ref':'LSRK'}
                                               })
    col13 = tableutil.makearrcoldesc('SYSVEL', 1.0, 1, 
                                     comment='Systemic velocity at reference', 
                                     keywords={'QuantumUnits':['m/s',], 
                                               'MEASINFO':{'type':'radialvelocity', 
                                                           'Ref':'LSRK'}
                                               })
    
    desc = tableutil.maketabdesc([col1, col2, col3, col4, col5, col6, col7, col8, col9, 
                                  col10, col11, col12, col13])
    tb = table("%s/SOURCE" % filename, desc, nrow=nint, ack=False)
    
    tb.putcol('DIRECTION', numpy.zeros((nint, 2)), 0, nint)
    tb.putcol('PROPER_MOTION', numpy.zeros((nint, 2)), 0, nint)
    tb.putcol('CALIBRATION_GROUP', [0,]*nint, 0, nint)
    tb.putcol('CODE', ['none',]*nint, 0, nint)
    tb.putcol('INTERVAL', [tint,]*nint, 0, nint)
    tb.putcol('NAME', ['zenith',]*nint, 0, nint)
    tb.putcol('NUM_LINES', [0,]*nint, 0, nint)
    tb.putcol('SOURCE_ID', list(range(nint)), 0, nint)
    tb.putcol('SPECTRAL_WINDOW_ID', [-1,]*nint, 0, nint)
    tb.putcol('TIME', [0.0,]*nint, 0, nint)
    #tb.putcol('TRANSITION', []*nint, 0, nint)
    #tb.putcol('REST_FREQUENCY', []*nint, 0, nint)
    #tb.putcol('SYSVEL', []*nint, 0, nint)
    
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
    # Field
    
    col1 = tableutil.makearrcoldesc('DELAY_DIR', 0.0, 2,
                                    comment='Direction of delay center (e.g. RA, DEC)as polynomial in time.', 
                                    keywords={'QuantumUnits':['rad','rad'], 
                                              'MEASINFO':{'type':'direction', 'Ref':'J2000'}
                                              })
    col2 = tableutil.makearrcoldesc('PHASE_DIR', 0.0, 2,
                                    comment='Direction of phase center (e.g. RA, DEC).', 
                                    keywords={'QuantumUnits':['rad','rad'], 
                                              'MEASINFO':{'type':'direction', 'Ref':'J2000'}
                                              })
    col3 = tableutil.makearrcoldesc('REFERENCE_DIR', 0.0, 2,
                                    comment='Direction of REFERENCE center (e.g. RA, DEC).as polynomial in time.', 
                                    keywords={'QuantumUnits':['rad','rad'], 
                                              'MEASINFO':{'type':'direction', 'Ref':'J2000'}
                                              })
    col4 = tableutil.makescacoldesc('CODE', "none", 
                                    comment='Special characteristics of field, e.g. Bandpass calibrator')
    col5 = tableutil.makescacoldesc('FLAG_ROW', False, 
                                    comment='Row Flag')
    col6 = tableutil.makescacoldesc('NAME', "none", 
                                    comment='Name of this field')
    col7 = tableutil.makescacoldesc('NUM_POLY', 0, 
                                    comment='Polynomial order of _DIR columns')
    col8 = tableutil.makescacoldesc('SOURCE_ID', 0, 
                                    comment='Source id')
    col9 = tableutil.makescacoldesc('TIME', 0.0, 
                                    comment='Time origin for direction and rate', 
                                    keywords={'QuantumUnits':['s',],
                                              'MEASINFO':{'type':'epoch', 'Ref':'UTC'}
                                              })
    
    desc = tableutil.maketabdesc([col1, col2, col3, col4, col5, col6, col7, col8, col9])
    tb = table("%s/FIELD" % filename, desc, nrow=nint, ack=False)
    
    tb.putcol('DELAY_DIR', numpy.zeros((nint, 1, 2)), 0, nint)
    tb.putcol('PHASE_DIR', numpy.zeros((nint, 1, 2)), 0, nint)
    tb.putcol('REFERENCE_DIR', numpy.zeros((nint, 1, 2)), 0, nint)
    tb.putcol('CODE', ['none',]*nint, 0, nint)
    tb.putcol('FLAG_ROW', [False,]*nint, 0, nint)
    tb.putcol('NAME', ['zenith',]*nint, 0, nint)
    tb.putcol('NUM_POLY', [0,]*nint, 0, nint)
    tb.putcol('SOURCE_ID', list(range(nint)), 0, nint)
    tb.putcol('TIME', [0.0,]*nint, 0, nint)
    
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
def _write_spectralwindow_table(filename, config):
    """
    Write the spectral window table.
    """
    
    station = config.station
    nant = config.nant
    nbl = config.nbl
    tint = config.tint
    freq = config.freq
    nchan = config.nchan
    chan_bw = config.chan_bw
    pols = config.pols
    npol = config.npol
    nint = config.nint
    
    # Spectral Window
    
    col1  = tableutil.makescacoldesc('MEAS_FREQ_REF', 0, 
                                     comment='Frequency Measure reference')
    col2  = tableutil.makearrcoldesc('CHAN_FREQ', 0.0, 1, 
                                     comment='Center frequencies for each channel in the data matrix', 
                                     keywords={'QuantumUnits':['Hz',], 
                                               'MEASINFO':{'type':'frequency', 
                                                           'VarRefCol':'MEAS_FREQ_REF', 
                                                           'TabRefTypes':['REST','LSRK','LSRD','BARY','GEO','TOPO','GALACTO','LGROUP','CMB','Undefined'],
                                                           'TabRefCodes':numpy.array([0,1,2,3,4,5,6,7,8,64], dtype=numpy.uint32)}
                                               })
    col3  = tableutil.makescacoldesc('REF_FREQUENCY', freq[0], 
                                     comment='The reference frequency', 
                                     keywords={'QuantumUnits':['Hz',], 
                                               'MEASINFO':{'type':'frequency', 
                                                           'VarRefCol':'MEAS_FREQ_REF', 
                                                           'TabRefTypes':['REST','LSRK','LSRD','BARY','GEO','TOPO','GALACTO','LGROUP','CMB','Undefined'],
                                                           'TabRefCodes':numpy.array([0,1,2,3,4,5,6,7,8,64], dtype=numpy.uint32)}
                                               })
    col4  = tableutil.makearrcoldesc('CHAN_WIDTH', 0.0, 1, 
                                     comment='Channel width for each channel', 
                                     keywords={'QuantumUnits':['Hz',]})
    col5  = tableutil.makearrcoldesc('EFFECTIVE_BW', 0.0, 1, 
                                     comment='Effective noise bandwidth of each channel', 
                                     keywords={'QuantumUnits':['Hz',]})
    col6  = tableutil.makearrcoldesc('RESOLUTION', 0.0, 1, 
                                     comment='The effective noise bandwidth for each channel', 
                                     keywords={'QuantumUnits':['Hz',]})
    col7  = tableutil.makescacoldesc('FLAG_ROW', False, 
                                     comment='flag')
    col8  = tableutil.makescacoldesc('FREQ_GROUP', 1, 
                                     comment='Frequency group')
    col9  = tableutil.makescacoldesc('FREQ_GROUP_NAME', "group1", 
                                     comment='Frequency group name')
    col10 = tableutil.makescacoldesc('IF_CONV_CHAIN', 0, 
                                     comment='The IF conversion chain number')
    col11 = tableutil.makescacoldesc('NAME', "%i channels" % nchan, 
                                     comment='Spectral window name')
    col12 = tableutil.makescacoldesc('NET_SIDEBAND', 0, 
                                     comment='Net sideband')
    col13 = tableutil.makescacoldesc('NUM_CHAN', 0, 
                                     comment='Number of spectral channels')
    col14 = tableutil.makescacoldesc('TOTAL_BANDWIDTH', 0.0, 
                                     comment='The total bandwidth for this window', 
                                     keywords={'QuantumUnits':['Hz',]})
    
    desc = tableutil.maketabdesc([col1, col2, col3, col4, col5, col6, col7, col8, col9, 
                                  col10, col11, col12, col13, col14])
    tb = table("%s/SPECTRAL_WINDOW" % filename, desc, nrow=1, ack=False)
    
    tb.putcell('MEAS_FREQ_REF', 0, 5)
    tb.putcell('CHAN_FREQ', 0, freq)
    tb.putcell('REF_FREQUENCY', 0, freq[0])
    tb.putcell('CHAN_WIDTH', 0, [chan_bw,]*nchan)
    tb.putcell('EFFECTIVE_BW', 0, [chan_bw,]*nchan)
    tb.putcell('RESOLUTION', 0, [chan_bw,]*nchan)
    tb.putcell('FLAG_ROW', 0, False)
    tb.putcell('FREQ_GROUP', 0, 1)
    tb.putcell('FREQ_GROUP_NAME', 0, 'group%i' % 1)
    tb.putcell('IF_CONV_CHAIN', 0, 0)
    tb.putcell('NAME', 0, "IF %i, %i channels" % (1, nchan))
    tb.putcell('NET_SIDEBAND', 0, 0)
    tb.putcell('NUM_CHAN', 0, nchan)
    tb.putcell('TOTAL_BANDWIDTH', 0, nchan*chan_bw)
    
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
   

def _write_misc_required_tables(filename, config): 
    station = config.station
    nant = config.nant
    nbl = config.nbl
    tint = config.tint
    freq = config.freq
    nchan = config.nchan
    pols = config.pols
    npol = config.npol
    nint = config.nint
    settings = config.settings
    
    # Flag command
    
    col1 = tableutil.makescacoldesc('TIME', 0.0, 
                                    comment='Midpoint of interval for which this flag is valid', 
                                    keywords={'QuantumUnits':['s',], 
                                              'MEASINFO':{'type':'epoch', 'Ref':'UTC'}
                                              })
    col2 = tableutil.makescacoldesc('INTERVAL', 0.0, 
                                    comment='Time interval for which this flag is valid', 
                                    keywords={'QuantumUnits':['s',]})
    col3 = tableutil.makescacoldesc('TYPE', 'flag', 
                                    comment='Type of flag (FLAG or UNFLAG)')
    col4 = tableutil.makescacoldesc('REASON', 'reason', 
                                    comment='Flag reason')
    col5 = tableutil.makescacoldesc('LEVEL', 0, 
                                    comment='Flag level - revision level')
    col6 = tableutil.makescacoldesc('SEVERITY', 0, 
                                    comment='Severity code (0-10)')
    col7 = tableutil.makescacoldesc('APPLIED', False, 
                                    comment='True if flag has been applied to main table')
    col8 = tableutil.makescacoldesc('COMMAND', 'command', 
                                    comment='Flagging command')
    
    desc = tableutil.maketabdesc([col1, col2, col3, col4, col5, col6, col7, col8])
    tb = table("%s/FLAG_CMD" % filename, desc, nrow=0, ack=False)
    
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
    # History
    
    col1 = tableutil.makescacoldesc('TIME', 0.0, 
                                    comment='Timestamp of message', 
                                    keywords={'QuantumUnits':['s',], 
                                              'MEASINFO':{'type':'epoch', 'Ref':'UTC'}
                                              })
    col2 = tableutil.makescacoldesc('OBSERVATION_ID', 0, 
                                    comment='Observation id (index in OBSERVATION table)')
    col3 = tableutil.makescacoldesc('MESSAGE', 'message', 
                                    comment='Log message')
    col4 = tableutil.makescacoldesc('PRIORITY', 'NORMAL', 
                                    comment='Message priority')
    col5 = tableutil.makescacoldesc('ORIGIN', 'origin', 
                                    comment='(Source code) origin from which message originated')
    col6 = tableutil.makescacoldesc('OBJECT_ID', 0, 
                                    comment='Originating ObjectID')
    col7 = tableutil.makescacoldesc('APPLICATION', 'application', 
                                    comment='Application name')
    col8 = tableutil.makearrcoldesc('CLI_COMMAND', 'command', 1, 
                                    comment='CLI command sequence')
    col9 = tableutil.makearrcoldesc('APP_PARAMS', 'params', 1, 
                                    comment='Application parameters')
    
    desc = tableutil.maketabdesc([col1, col2, col3, col4, col5, col6, col7, col8, col9])
    tb = table("%s/HISTORY" % filename, desc, nrow=0, ack=False)
    
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
    # POINTING
    
    col1 = tableutil.makescacoldesc('ANTENNA_ID', 0, 
                                    comment='Antenna Id')
    col2 = tableutil.makescacoldesc('TIME', 0.0, 
                                    comment='Time interval midpoint', 
                                    keywords={'QuantumUnits':['s',], 
                                              'MEASINFO':{'type':'epoch', 'Ref':'UTC'}
                                              })
    col3 = tableutil.makescacoldesc('INTERVAL', 0.0, 
                                    comment='Time interval', 
                                    keywords={'QuantumUnits':['s',]})
    col4 = tableutil.makescacoldesc('NAME', 'name', 
                                    comment='Pointing position name')
    col5 = tableutil.makescacoldesc('NUM_POLY', 0, 
                                    comment='Series order')
    col6 = tableutil.makescacoldesc('TIME_ORIGIN', 0.0, 
                                    comment='Time origin for direction', 
                                    keywords={'QuantumUnits':['s',], 
                                              'MEASINFO':{'type':'epoch', 'Ref':'UTC'}
                                              })
    col7 = tableutil.makearrcoldesc('DIRECTION', 0.0, 2, 
                                    comment='Antenna pointing direction as polynomial in time', 
                                    keywords={'QuantumUnits':['rad','rad'], 
                                              'MEASINFO':{'type':'direction', 'Ref':'J2000'}
                                              })
    col8 = tableutil.makearrcoldesc('TARGET', 0.0, 2, 
                                    comment='target direction as polynomial in time',
                                    keywords={'QuantumUnits':['rad','rad'], 
                                              'MEASINFO':{'type':'direction', 'Ref':'J2000'}
                                              })
    col9 = tableutil.makescacoldesc('TRACKING', True, 
                                    comment='Tracking flag - True if on position')
    
    desc = tableutil.maketabdesc([col1, col2, col3, col4, col5, col6, col7, col8, col9])
    tb = table("%s/POINTING" % filename, desc, nrow=0, ack=False)
    
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
    # Processor
    
    col1 = tableutil.makescacoldesc('TYPE', 'type', 
                                    comment='Processor type')
    col2 = tableutil.makescacoldesc('SUB_TYPE', 'subtype', 
                                    comment='Processor sub type')
    col3 = tableutil.makescacoldesc('TYPE_ID', 0, 
                                    comment='Processor type id')
    col4 = tableutil.makescacoldesc('MODE_ID', 0, 
                                    comment='Processor mode id')
    col5 = tableutil.makescacoldesc('FLAG_ROW', False, 
                                    comment='flag')
    col6 = tableutil.makescacoldesc('SETTINGS', settings, 
                                    comment='ARX and F-engine settings')
    
    desc = tableutil.maketabdesc([col1, col2, col3, col4, col5, col6])
    tb = table("%s/PROCESSOR" % filename, desc, nrow=0, ack=False)
    
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
    # State
    
    col1 = tableutil.makescacoldesc('SIG', True, 
                                    comment='True for a source observation')
    col2 = tableutil.makescacoldesc('REF', False, 
                                    comment='True for a reference observation')
    col3 = tableutil.makescacoldesc('CAL', 0.0, 
                                    comment='Noise calibration temperature', 
                                    keywords={'QuantumUnits':['K',]})
    col4 = tableutil.makescacoldesc('LOAD', 0.0, 
                                    comment='Load temperature', 
                                    keywords={'QuantumUnits':['K',]})
    col5 = tableutil.makescacoldesc('SUB_SCAN', 0, 
                                    comment='Sub scan number, relative to scan number')
    col6 = tableutil.makescacoldesc('OBS_MODE', 'mode', 
                                    comment='Observing mode, e.g., OFF_SPECTRUM')
    col7 = tableutil.makescacoldesc('FLAG_ROW', False, 
                                    comment='Row flag')
    
    desc = tableutil.maketabdesc([col1, col2, col3, col4, col5, col6, col7])
    tb = table("%s/STATE" % filename, desc, nrow=0, ack=False)
    
    if FORCE_TABLE_FLUSH:
        tb.flush()
    tb.close()
    
