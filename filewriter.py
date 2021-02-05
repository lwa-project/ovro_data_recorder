import os
import sys
import h5py
import json
import shutil
import subprocess
from datetime import datetime

from common import FS, CLOCK, CHAN_BW, chan_to_freq, timetag_to_datetime, timetag_to_tuple


class FileWriterBase(object):
    def __init__(self, filename, start_time, stop_time):
        self.filename = filename
        self.start_time = start_time
        self.stop_time = stop_time
        
        self._started = False
        self._interface = None
        
    @property
    def is_active(self):
        now = datetime.utcnow()
        return ((now >= self.start_time) and (now <= self.stop_time))
        
    @property
    def is_started(self):
        return self._started
        
    @property
    def is_expired(self):
        now = datetime.utcnow()
        return now > self.stop_time
        
    @property
    def size(self):
        filesize = None
        if os.path.exists(self.filename):
            filesize = os.path.getsize(self.filename)
        return filesize
        
    @property
    def mtime(self):
        filemtime = None
        if os.path.exists(self.filename):
            filemtime = os.path.getmtime(self.filename)
        return filemtime
        
    def start(self, *args, **kwds):
        raise NotImplementedError
        
    def write(self, time_tag, data, **kwds):
        raise NotImplementedError
        
    def stop(self):
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
        raise NotImplementedError


class TarredFileWriterBase(FileWriterBase):
    def post_stop_task(self):
        subprocess.check_output(['tar', 'czvf', self.filename+'.tar.gz', self.filename])


class HDF5Writer(FileWriterBase):
    def start(self, beam, chan0, navg, nchan, npol, pols, **kwds):
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
        obs.attrs['RBW'] = CHAN_BW
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
            
        frequency = numpy.arange(nchan)*CHAN_BW + chan_to_freq(chan0)    
        grp['freq'] = frequency.astype(numpy.float64)
        grp['freq'].attrs['Units'] = 'Hz'
        
        if isinstance(pols, str):
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
        if not self.is_active:
            return False
            
        size = data.shape[0]
        self._time[self._counter:self._counter+size] = [timetag_to_tuple(time_tag+i*self._time_step) for i in range(size)]
        for i in range(data.shape[-1]):
            self._pols[i][self._counter:self._counter+size,:] = data[...,i]
        self._counter += size


class MeasurementSetWriter(FileWriterBase):
    def start(self, antennas):
        self.tempdir = '/dev/shm/%s-%i' % (type(self).__name__, os.getpid())
        if not os.path.exists(self.tempdir):
            os.mkdir(self.tempdir)
            
        # Save
        self._antennas = antennas
        self._started = True
        
    def write(self, time_tag, data):
        dt = timetag_to_datetime(time_tag)
        
        tempname = os.path.join(self.tempdir, dt.strftime('%Y%m%d_%H%M%S'))
        
        # TODO
        # TODO
        # TODO
        # TODO
        
        filename = "%s_%s.tar.hz" % (self.filename, dt.strftime('%Y%m%d_%H%M%S'))
        subprocess.check_call(['tar', 'czvf', filename, tempname], cwd=self.tempdir)
        shutil.rmtree(tempname)
