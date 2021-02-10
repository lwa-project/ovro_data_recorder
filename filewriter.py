import os
import sys
import h5py
import json
import numpy
import shutil
import subprocess
from datetime import datetime
from textwrap import fill as tw_fill

from casacore.tables import table, tableutil

from common import FS, CLOCK, NCHAN, CHAN_BW, chan_to_freq, timetag_to_datetime, timetag_to_tuple, timetag_to_astropy


# Temporary file directory
TEMP_BASEDIR = "/dev/shm"


# Measurement set stokes name -> number
STOKES_CODES = {'I': 1,  'Q': 2,  'U': 3,  'V': 4, 
               'RR': 5, 'RL': 6, 'LR': 7, 'LL': 8,
               'XX': 9, 'XY':10, 'YX':11, 'YY':12}
               

# Measurement set stokes number -> name
NUMERIC_STOKES = { 1:'I',   2:'Q',   3:'U',   4:'V', 
                   5:'RR',  6:'RL',  7:'LR',  8:'LL',
                   9:'XX', 10:'XY', 11:'YX', 12:'YY'}


class FileWriterBase(object):
    """
    Class to represent a file to write data to for the specified time period.
    """
    
    def __init__(self, filename, start_time, stop_time, reduction=None):
        self.filename = filename
        self.start_time = start_time
        self.stop_time = stop_time
        self.reduction = None
        
        self._started = False
        self._interface = None
        
    def __repr__(self):
        output = "<%s filename='%s', start_time='%s', stop_time='%s', reduction=%s>" % (type(self).__name__,
                                                                                        self.filename,
                                                                                        self.start_time,
                                                                                        self.stop_time,
                                                                                        self.reduction)
        return tw_fill(output, subsequent_indent='    ')
        
    @property
    def is_active(self):
        """
        Whether or not the file should be considered active, i.e., the current
        time is within its scheduled window.
        """
        
        now = datetime.utcnow()
        return ((now >= self.start_time) and (now <= self.stop_time))
        
    @property
    def is_started(self):
        """
        Whether or not the file as been started.
        """
        
        return self._started
        
    @property
    def is_expired(self):
        """
        Whether or not the file is expired, i.e., the current time is past the
        file's stop time.
        """
        
        now = datetime.utcnow()
        return now > self.stop_time
        
    @property
    def size(self):
        """
        The current size of the file or None if the file does not exist yet.
        """
        
        filesize = None
        if os.path.exists(self.filename):
            filesize = os.path.getsize(self.filename)
        return filesize
        
    @property
    def mtime(self):
        """
        The current modifiction time fo the file or None if the file does not
        exist yet.
        """
        
        filemtime = None
        if os.path.exists(self.filename):
            filemtime = os.path.getmtime(self.filename)
        return filemtime
        
    def start(self, *args, **kwds):
        """
        Method to call when starting the file that initializes all of the file's
        metadata.  To be overridden by sub-classes.
        """
        
        raise NotImplementedError
        
    def write(self, time_tag, data, **kwds):
        """
        Method to call when writing data to a file that has been started.  To
        be overridden by sub-classes.
        """
        
        raise NotImplementedError
        
    def stop(self):
        """
        Close out the file and then call the 'post_stop_task' method.
        """
        
        try:
            self._interface.close()
        except AttributeError:
            pass
            
        if os.path.exists(self.filename):
            try:
                self.post_stop_task()
            except NotImplementedError:
                pass
                
    def post_stop_task(self):
        """
        Method to preform any tasks that are needed after a file as stopped.
        """
        
        raise NotImplementedError
        
    def cancel(self):
        """
        Cancel the file and stop any current writing to it.
        """
        
        if self.is_active:
            self.stop()
        self.stop = datetime.utcnow()


class TarredFileWriterBase(FileWriterBase):
    """
    Sub-class of FileWriterBase that wraps the output file in a gzipped tar file
    after the writing has stopped.
    """
    
    def post_stop_task(self):
        subprocess.check_output(['tar', 'czvf', self.filename+'.tar.gz', self.filename])


class HDF5Writer(FileWriterBase):
    """
    Sub-class of FileWriterBase that writes data to a HDF5 file.
    """
    
    def start(self, beam, chan0, navg, nchan, chan_bw, npol, pols, **kwds):
        """
        Set the metadata in the HDF5 file and prepare it for writing.
        """
        
        f = h5py.File(self.filename)
        
        # File structure
        chunks = int((self.stop_time - self.start_time).total_seconds() / (navg / CHAN_BW))
        
        # Top level attributes
        ## Observer and Project Info.
        f.attrs['ObserverID'] = 0
        f.attrs['ObserverName'] = ''
        f.attrs['ProjectID'] = ''
        f.attrs['SessionID'] = 0
        
        ## Station information
        f.attrs['StationName'] = 'ovrolwa'
        
        ## File creation time
        f.attrs['FileCreation'] = datetime.utcnow().strftime("UTC %Y/%m/%d %H:%M:%S")
        f.attrs['FileGenerator'] = ''
        
        ## Input file info.
        f.attrs['InputMetadata'] = json.dumps(kwds)
        
        # Observation group
        ## Get the group or create it if it doesn't exist
        obs = f.get('/Observation1', None)
        if obs is None:
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
        obs.attrs['tInt'] = navg / CHAN_BW
        obs.attrs['tInt_Units'] = 's'
        obs.attrs['LFFT'] = NCHAN
        obs.attrs['nChan'] = nchan
        obs.attrs['RBW'] = chan_bw
        obs.attrs['RBW_Units'] = 'Hz'
        
        # Data structures
        ## Time
        if 'time' not in obs:
            t = obs.create_dataset('time', (chunks,), dtype=numpy.dtype({"names": ["int", "frac"],
                                                                         "formats": ["i8", "f8"]}))
            t.attrs['format'] = 'unix'
            t.attrs['scale'] = 'utc'
            
        ## The "tuning"
        grp = obs.get('Tuning1', None)
        if grp is None:
            grp = obs.create_group('Tuning1')
            
        frequency = numpy.arange(nchan)*chan_bw+ chan_to_freq(chan0)
        grp['freq'] = frequency.astype(numpy.float64)
        grp['freq'].attrs['Units'] = 'Hz'
        
        if not isinstance(pols, (tuple, list)):
            pols = [p.strip().rstrip() for p in pols.split(',')]
        data_products = {}
        for i,p in enumerate(pols):
            d = grp.create_dataset(p, (chunks, frequency.size), 'f4')
            d.attrs['axis0'] = 'time'
            d.attrs['axis1'] = 'frequency'
            data_products[i] = d
            data_products[p] = d
            
        # Save
        self._iterface = f
        self._obs = obs
        self._time = t
        self._time_step = navg * (int(FS) / int(CHAN_BW))
        self._pols = data_products
        self._counter = 0
        self._started = True
        
    def write(self, time_tag, data):
        """
        Write a collection of dynamic spectra to the HDF5 file.
        """
        
        if not self.is_active:
            return False
        elif not self.is_started:
            raise RuntimeError("File is active but has not be started")
            
        # Find what integrations fit within the file's window
        size = min([self._time.size-self._counter, data.shape[0]])
        # Timestamps
        self._time[self._counter:self._counter+size] = [timetag_to_tuple(time_tag+i*self._time_step) for i in range(size)]
        # The data
        for i in range(data.shape[-1]):
            self._pols[i][self._counter:self._counter+size,:] = data[:size,0,:,i]
        # Update the counter
        self._counter += size


class MeasurementSetWriter(FileWriterBase):
    """
    Sub-class of FileWriterBase that writes data to a measurement set.  Each
    call to write leads to a new measurement set.
    """
    
    def start(self, station, chan0, navg, nchan, chan_bw, npol, pols):
        """
        Set the metadata for the measurement sets and create the template.
        """
        
        self._tempdir = os.path.join(TEMP_BASEDIR, '%s-%i' % (type(self).__name__, os.getpid()))
        if not os.path.exists(self._tempdir):
            os.mkdir(self._tempdir)
            
        # Save
        self._station = station
        self._tint = navg / CHAN_BW
        self._time_step = navg * (int(FS) / int(CHAN_BW))
        self._nant = len(self._station.antennas)
        self._freq = numpy.arange(nchan)*chan_bw+ chan_to_freq(chan0)
        self._nchan = nchan
        self._pols = [STOKES_CODES[p] for p in pols]
        self._npol = len(self._pols)
        self._nbl = self._nant*(self._nant + 1) // 2
        self._started = True
        
        # Create the template
        self._template = self._create_template()
        
    def _create_template(self):
        """
        Create a template measurement set with the right set of tables, minus
        main.
        """
        
        self._write_antenna_table()
        self._write_polarization_table()
        self._write_observation_table()
        self._write_spectralwindow_table()
        self._write_misc_required_tables()
        
    def _write_main_table(self):
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
        tb = table("%s" % self.basename, desc, nrow=0, ack=False)
        
        
        fg = numpy.zeros((self._nbl,self._npol,self._nchan), dtype=numpy.bool)
        fc = numpy.zeros((self._nbl,self._npol,self._nchan,1), dtype=numpy.bool)
        uv = numpy.zeros((self._nbl,3), dtype=numpy.float64)
        a1 = numpy.zeros((self._nbl,), dtype=numpy.int32)
        a2 = numpy.zeros((self._nbl,), dtype=numpy.int32)
        vs = numpy.zeros((self._nbl,self._npol,self._nchan), dtype=numpy.complex64)
        wg = numpy.ones((self._nbl,self._npol))
        sg = numpy.ones((self._nbl,self._npol))*9999
        
        k = 0
        for i in range(self._nant):
            l1 = self._station.antennas[i]
            for j in range(i, self._nant):
                l2 = self._station.antennas[j]
                
                uv[k,:] = (l1[0]-l2[0], l1[1]-l2[1], l1[2]-l2[2])
                a1[k] = i
                a2[k] = j
                
        tb.putcol('UVW', uvwList, 0, self._nbl)
        tb.putcol('FLAG', fg.transpose(0,2,1), 0, self._nbl)
        tb.putcol('FLAG_CATEGORY', fc.transpose(0,3,2,1), 0, self._nbl)
        tb.putcol('WEIGHT', wg, 0, self._nbl)
        tb.putcol('SIGMA', sg, 0, self._nbl)
        tb.putcol('ANTENNA1', a1, 0, self._nbl)
        tb.putcol('ANTENNA2', a2, 0, self._nbl)
        tb.putcol('ARRAY_ID', [0,]*self._nbl, 0, self._nbl)
        tb.putcol('DATA_DESC_ID', [0,]*self._nbl, 0, self._nbl)
        tb.putcol('EXPOSURE', [self._tint,]*self._nbl, 0, self._nbl)
        tb.putcol('FEED1', [0,]*self._nbl, 0, self._nbl)
        tb.putcol('FEED2', [0,]*self._nbl, 0, self._nbl)
        tb.putcol('FIELD_ID', [0,]*self._nbl, 0, self._nbl)
        tb.putcol('FLAG_ROW', [False,]*self._nbl, 0, self._nbl)
        tb.putcol('INTERVAL', [self._tint,]*self._nbl, 0, self._nbl)
        tb.putcol('OBSERVATION_ID', [0,]*self._nbl, 0, self._nbl)
        tb.putcol('PROCESSOR_ID', [-1,]*self._nbl, 0, self._nbl)
        tb.putcol('SCAN_NUMBER', [1,]*self._nbl, 0, self._nbl)
        tb.putcol('STATE_ID', [-1,]*self._nbl, 0, self._nbl)
        tb.putcol('TIME', [0.0,]*self._nbl, 0, self._nbl)
        tb.putcol('TIME_CENTROID', [0.0,]*self._nbl, 0, self._nbl)
        tb.putcol('DATA', vs.transpose(0,2,1), 0, self._nbl)
        
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
        tb = table("%s/DATA_DESCRIPTION" % self.basename, desc, nrow=1, ack=False)
        
        tb.putcell('FLAG_ROW', 0, False)
        tb.putcell('POLARIZATION_ID', 0, 0)
        tb.putcell('SPECTRAL_WINDOW_ID', 0, 0)
        
        tb.flush()
        tb.close()
        
    def _write_antenna_table(self):
        """
        Write the antenna table.
        """
        
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
        col8 = tableutil.makescacoldesc('STATION', self._station.name, 
                                        comment='Station (antenna pad) name')
        
        desc = tableutil.maketabdesc([col1, col2, col3, col4, col5, col6, col7, col8])
        tb = table("%s/ANTENNA" % self._template, desc, nrow=self._nant, ack=False)
        
        tb.putcol('OFFSET', numpy.zeros((self._nant,3)), 0, self._nant)
        tb.putcol('TYPE', ['GROUND-BASED,']*self._nant, 0, self._nant)
        tb.putcol('DISH_DIAMETER', [2.0,]*self._nant, 0, self._nant)
        tb.putcol('FLAG_ROW', [False,]*self._nant, 0, self._nant)
        tb.putcol('MOUNT', ['ALT-AZ',]*self._nant, 0, self._nant)
        tb.putcol('NAME', [ant.get_name() for ant in self.array[0]['ants']], 0, self._nant)
        tb.putcol('STATION', [self._station.name,]*self._nant, 0, self._nant)
        
        for i,ant in enumerate(self._station.antennas):
            #tb.putcell('OFFSET', i, [0.0, 0.0, 0.0])
            tb.putcell('POSITION', i, ant.ecef)
            #tb.putcell('TYPE', i, 'GROUND-BASED')
            #tb.putcell('DISH_DIAMETER', i, 2.0)
            #tb.putcell('FLAG_ROW', i, False)
            #tb.putcell('MOUNT', i, 'ALT-AZ')
            #tb.putcell('NAME', i, ant.get_name())
            #tb.putcell('STATION', i, self._station.name)
            
        tb.flush()
        tb.close()
        
    def _write_polarization_table(self):
        """
        Write the polarization table.
        """
        
        # Polarization
        
        stks = numpy.array(self._pols)
        prds = numpy.zeros((2,self._npol), dtype=numpy.int32)
        for i,stk in enumerate(self._pols):
            stks[i] = stk
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
        col4 = tableutil.makescacoldesc('NUM_CORR', self._npol, 
                                        comment='Number of correlation products')
        
        desc = tableutil.maketabdesc([col1, col2, col3, col4])
        tb = table("%s/POLARIZATION" % self._template, desc, nrow=1, ack=False)
        
        tb.putcell('CORR_TYPE', 0, self._pols)
        tb.putcell('CORR_PRODUCT', 0, prds.T)
        tb.putcell('FLAG_ROW', 0, False)
        tb.putcell('NUM_CORR', 0, self._npol)
        
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
        tb = table("%s/FEED" % self._template, desc, nrow=self._nant, ack=False)
        
        presp = numpy.zeros((self._nant,2,2), dtype=numpy.complex64)
        if self._pols[0] > 8:
            ptype = numpy.tile(['X', 'Y'], (self._nant,1))
            presp[:,0,0] = 1.0
            presp[:,0,1] = 0.0
            presp[:,1,0] = 0.0
            presp[:,1,1] = 1.0
        elif self._pols[0] > 4:
            ptype = numpy.tile(['R', 'L'], (self._nant,1))
            presp[:,0,0] = 1.0
            presp[:,0,1] = -1.0j
            presp[:,1,0] = 1.0j
            presp[:,1,1] = 1.0
        else:
            ptype = numpy.tile(['X', 'Y'], (self._nant,1))
            presp[:,0,0] = 1.0
            presp[:,0,1] = 0.0
            presp[:,1,0] = 0.0
            presp[:,1,1] = 1.0
            
        tb.putcol('POSITION', numpy.zeros((self._nant,3)), 0, self._nant)
        tb.putcol('BEAM_OFFSET', numpy.zeros((self._nant,2,2)), 0, self._nant)
        tb.putcol('POLARIZATION_TYPE', ptype, 0, self._nant)
        tb.putcol('POL_RESPONSE', presp, 0, self._nant)
        tb.putcol('RECEPTOR_ANGLE', numpy.zeros((self._nant,2)), 0, self._nant)
        tb.putcol('ANTENNA_ID', list(range(self._nant)), 0, self._nant)
        tb.putcol('BEAM_ID', [-1,]*self._nant, 0, self._nant)
        tb.putcol('FEED_ID', [0,]*self._nant, 0, self._nant)
        tb.putcol('INTERVAL', [self._tint,]*self._nant, 0, self._nant)
        tb.putcol('NUM_RECEPTORS', [2,]*self._nant, 0, self._nant)
        tb.putcol('SPECTRAL_WINDOW_ID', [-1,]*self._nant, 0, self._nant)
        tb.putcol('TIME', [0.0,]*self._nant, 0, self._nant)
        
        tb.flush()
        tb.close()
        
    def _write_observation_table(self):
        """
        Write the observation table.
        """
        
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
        col5 = tableutil.makescacoldesc('OBSERVER', self.observer, 
                                        comment='Name of observer(s)')
        col6 = tableutil.makescacoldesc('PROJECT', self.project, 
                                        comment='Project identification string')
        col7 = tableutil.makescacoldesc('RELEASE_DATE', 0.0, 
                                        comment='Release date when data becomes public', 
                                        keywords={'QuantumUnits':['s',], 
                                                  'MEASINFO':{'type':'epoch', 'Ref':'UTC'}
                                                  })
        col8 = tableutil.makescacoldesc('SCHEDULE_TYPE', self.mode, 
                                        comment='Observing schedule type')
        col9 = tableutil.makescacoldesc('TELESCOPE_NAME', self._station.name, 
                                        comment='Telescope Name (e.g. WSRT, VLBA)')
        
        desc = tableutil.maketabdesc([col1, col2, col3, col4, col5, col6, col7, col8, col9])
        tb = table("%s/OBSERVATION" % self._template, desc, nrow=1, ack=False)
        
        tb.putcell('TIME_RANGE', 0, [0.0, 0.0])
        tb.putcell('LOG', 0, 'Not provided')
        tb.putcell('SCHEDULE', 0, 'Not provided')
        tb.putcell('FLAG_ROW', 0, False)
        tb.putcell('OBSERVER', 0, self._station.name)
        tb.putcell('PROJECT', 0, self._station.name)
        tb.putcell('RELEASE_DATE', 0, 0.0)
        tb.putcell('SCHEDULE_TYPE', 0, type(self).__name__)
        tb.putcell('TELESCOPE_NAME', 0, self._station.name)
        
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
        tb = table("%s/SOURCE" % self._template, desc, nrow=1, ack=False)
        
        tb.putcell('DIRECTION', 0, [0.0, 0.0])
        tb.putcell('PROPER_MOTION', 0, [0.0, 0.0])
        tb.putcell('CALIBRATION_GROUP', 0, 0)
        tb.putcell('CODE', 0, 'none')
        tb.putcell('INTERVAL', self._tint, 0.0)
        tb.putcell('NAME', 0, 'zenith')
        tb.putcell('NUM_LINES', 0, 0)
        tb.putcell('SOURCE_ID', 0, i)
        tb.putcell('SPECTRAL_WINDOW_ID', 0, -1)
        tb.putcell('TIME', 0, 0.0)
        #tb.putcell('TRANSITION', 0, [])
        #tb.putcell('REST_FREQUENCY', 0, [])
        #tb.putcell('SYSVEL', 0, [])
        
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
        tb = table("%s/FIELD" % self._template, desc, nrow=1, ack=False)
        
        tb.putcell('DELAY_DIR', 0, numpy.array([[0.0, 0.0],]))
        tb.putcell('PHASE_DIR', 0, numpy.array([[0.0, 0.0],]))
        tb.putcell('REFERENCE_DIR', 0, numpy.array([[0.0, 0.0],]))
        tb.putcell('CODE', 0, 'None')
        tb.putcell('FLAG_ROW', 0, False)
        tb.putcell('NAME', 0, 'zenith')
        tb.putcell('NUM_POLY', 0, 0)
        tb.putcell('SOURCE_ID', 0, 0)
        tb.putcell('TIME', 0, 0.0)
        
        tb.flush()
        tb.close()
        
    def _write_spectralwindow_table(self):
        """
        Write the spectral window table.
        """
        
        # Spectral Window
        
        col1  = tableutil.makescacoldesc('MEAS_FREQ_REF', 0, 
                                         comment='Frequency Measure reference')
        col2  = tableutil.makearrcoldesc('CHAN_FREQ', 0.0, 1, 
                                         comment='Center frequencies for each channel in the data matrix', 
                                         keywords={'QuantumUnits':['Hz',], 
                                                   'MEASINFO':{'type':'frequency', 
                                                               'VarRefCol':'MEAS_FREQ_REF', 
                                                               'TabRefTypes':['REST','LSRK','LSRD','BARY','GEO','TOPO','GALACTO','LGROUP','CMB','Undefined'],
                                                               'TabRefCodes':[0,1,2,3,4,5,6,7,8,64]}
                                                   })
        col3  = tableutil.makescacoldesc('REF_FREQUENCY', self._freq[0], 
                                         comment='The reference frequency', 
                                         keywords={'QuantumUnits':['Hz',], 
                                                   'MEASINFO':{'type':'frequency', 
                                                               'VarRefCol':'MEAS_FREQ_REF', 
                                                               'TabRefTypes':['REST','LSRK','LSRD','BARY','GEO','TOPO','GALACTO','LGROUP','CMB','Undefined'],
                                                               'TabRefCodes':[0,1,2,3,4,5,6,7,8,64]}
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
        col11 = tableutil.makescacoldesc('NAME', "%i channels" % self._nchan, 
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
        tb = table("%s/SPECTRAL_WINDOW" % self._template, desc, nrow=1, ack=False)
        
        tb.putcell('MEAS_FREQ_REF', 0, 0)
        tb.putcell('CHAN_FREQ', 0, self._freq)
        tb.putcell('REF_FREQUENCY', 0, self._freq[0])
        tb.putcell('CHAN_WIDTH', 0, [self._freq[1]-self._freq[0],]*self._nchan)
        tb.putcell('EFFECTIVE_BW', 0, [self._freq[1]-self._freq[0],]*self._nchan)
        tb.putcell('RESOLUTION', 0, [self._freq[1]-self._freq[0],]*self._nchan)
        tb.putcell('FLAG_ROW', 0, False)
        tb.putcell('FREQ_GROUP', 0, 1)
        tb.putcell('FREQ_GROUP_NAME', 0, 'group%i' % 1)
        tb.putcell('IF_CONV_CHAIN', 0, 0)
        tb.putcell('NAME', 0, "IF %i, %i channels" % (1, self._nchan))
        tb.putcell('NET_SIDEBAND', 0, 0)
        tb.putcell('NUM_CHAN', 0, self._nchan)
        tb.putcell('TOTAL_BANDWIDTH', 0, self._nchan*(self._freq[1]-self._freq[0]))
        
        tb.flush()
        tb.close()
        
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
        tb = table("%s/FLAG_CMD" % self._template, desc, nrow=0, ack=False)
        
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
        tb = table("%s/HISTORY" % self._template, desc, nrow=0, ack=False)
        
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
        tb = table("%s/POINTING" % self._template, desc, nrow=0, ack=False)
        
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
        
        desc = tableutil.maketabdesc([col1, col2, col3, col4, col5])
        tb = table("%s/PROCESSOR" % self._template, desc, nrow=0, ack=False)
        
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
        tb = table("%s/STATE" % self._template, desc, nrow=0, ack=False)
        
        tb.flush()
        tb.close()
        
    def write(self, time_tag, data):
        dt = timetag_to_datetime(time_tag)
        ap = timetag_to_astropy(time_tag)
        ct = timetag_to_astropy(time_tag + self._time_step // 2)
        ed = timetag_to_astropy(time_tag + self._time_step)
        
        # Make a copy of the template
        tempname = os.path.join(self._tempdir, dt.strftime('%Y%m%d_%H%M%S'))
        subprocess.check_call(['cp', '-r', self._template, tempname])
        
        # Find the point overhead
        zen = [ct.sidereal_time('mean', self._station.astropy).to_value('rad'),
               self._station.lat*numpy.pi/180]
        
        # Fill in the main table
        tb = table(tempname, ack=False)
        tb.putcol('DATA', data.transpose(0,2,1), 0, self._nbl)
        tb.putcol('TIME', [ap.mjd,]*self._nbl, 0, self._nbl)
        tb.putcol('TIME_CENTROID',[ct.mjd,]*self._nbl, 0, self._nbl)
        tb.flush()
        tb.close()
        
        # Update the feed table
        tb = table(os.path.join(tempname, "FEED"), ack=False)
        tb.putcol('TIME', [ap.mjd,]*self._nbl, 0, 1)
        tb.flush()
        tb.close()
        
        # Update the observation table
        tb = table(os.path.join(tempname, "OBSERVATION"), ack=False)
        tb.putcol('TIME_RANGE', [[ap.mjd,ed.mjd],]*self._nbl, 0, 1)
        tb.flush()
        tb.close()
        
        # Update the source table
        tb = table(os.path.join(tempname, "SOURCE"), ack=False)
        tb.putcol('TIME', ap.mjd, 0, 1)
        tb.putcol('DIRECTION', zen, 0, 1)
        tb.flush()
        tb.close()
        
        # Update the field table
        tb = table(os.path.join(tempname, "FIELD"), ack=False)
        tb.putcol('TIME', ap.mjd, 0, 1)
        tb.putcol('DELAY_DIR', zen, 0, 1)
        tb.putcol('PHASE_DIR', zen, 0, 1)
        tb.putcol('REFERENCE_DIR', zen, 0, 1)
        tb.flush()
        tb.close()
        
        # Save it to its final location
        filename = "%s_%s.tar.gz" % (self.filename, dt.strftime('%Y%m%d_%H%M%S'))
        subprocess.check_call(['tar', 'czvf', filename, tempname], cwd=self._tempdir)
        shutil.rmtree(tempname)
        
    def stop(self):
        """
        Close out the file and then call the 'post_stop_task' method.
        """
        
        shutil.rmtree(self._template)
        try:
            os.rmdir(self.tempdir)
        except OSError:
            pass
            
        try:
            self.post_stop_task()
        except NotImplementedError:
            pass
