import os
import sys
import h5py
import json
import numpy
import atexit
import shutil
import subprocess
from datetime import datetime, timedelta
from textwrap import fill as tw_fill

from casacore.tables import table, tableutil

from common import FS, CLOCK, NCHAN, CHAN_BW, chan_to_freq, timetag_to_datetime, timetag_to_tuple, timetag_to_astropy
from lwahdf import *
from lwams import *

# Temporary file directory
TEMP_BASEDIR = "/dev/shm"


class FileWriterBase(object):
    """
    Class to represent a file to write data to for the specified time period.
    """
    
    def __init__(self, filename, start_time, stop_time, reduction=None):
        self.filename = os.path.abspath(filename)
        self.start_time = start_time
        self.stop_time = stop_time
        self.reduction = reduction
        
        self._queue = None
        self._started = False
        self._interface = None
        
    def __repr__(self):
        output = "<%s filename='%s', start_time='%s', stop_time='%s', reduction=%s>" % (type(self).__name__,
                                                                                        self.filename,
                                                                                        self.start_time,
                                                                                        self.stop_time,
                                                                                        self.reduction)
        return tw_fill(output, subsequent_indent='    ')
        
    def utcnow(self):
        now = datetime.utcnow()
        if self._queue is not None:
            now = now - self._queue.lag
        return now
        
    @property
    def is_pending(self):
        """
        Whether or not the file should be considered pending, i.e., the current
        time is within one second before its scheduled window starts.
        """
        
        nowish = self.utcnow() + timedelta(seconds=1)
        return nowish >= self.start_time
    
    @property
    def is_active(self):
        """
        Whether or not the file should be considered active, i.e., the current
        time is within its scheduled window.
        """
        
        now = self.utcnow()
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
        
        now = self.utcnow()
        return now > self.stop_time
        
    @property
    def is_post_pending(self):
        """
        Whether or not the file should be considered post-pending, i.e., the
        current time is within one second after its scheduled window stops.
        """
        
        nowish = self.utcnow() - timedelta(seconds=1)
        return nowish <= self.stop_time
        
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
        with open('/dev/null', 'wb') as dn:
            subprocess.check_output(['tar', 'czf', self.filename+'.tar.gz', self.filename],
                                     stderr=dn)


class HDF5Writer(FileWriterBase):
    """
    Sub-class of FileWriterBase that writes data to a HDF5 file.
    """
    
    def start(self, beam, chan0, navg, nchan, chan_bw, npol, pols, **kwds):
        """
        Set the metadata in the HDF5 file and prepare it for writing.
        """
        
        # Reduction adjustments
        freq = numpy.arange(nchan)*chan_bw + chan_to_freq(chan0)
        if self.reduction is not None:
            navg = navg * self.reduction.reductions[0]
            nchan = nchan // self.reduction.reductions[2]
            chan_bw = chan_bw * self.reduction.reductions[2]
            pols = self.reduction.pols
            
            freq = freq.reshape(-1, self.reduction.reductions[2])
            freq = freq.mean(axis=1)
            
        # Expected integration count
        chunks = int((self.stop_time - self.start_time).total_seconds() / (navg / CHAN_BW))
        
        # Create and fill
        self._interface = create_hdf5(self.filename, beam)
        set_frequencies(self._interface, freq)
        self._time = set_time(self._interface, navg / CHAN_BW, chunks)
        self._time_step = navg * (int(FS) / int(CHAN_BW))
        self._pols = set_data_products(self._interface, pols, chunks)
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
            
        # Reduction
        if self.reduction is not None:
            data = self.reduction(data)
            
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
    
    def __init__(self, filename, start_time, stop_time):
        FileWriterBase.__init__(self, filename, start_time, stop_time, reduction=None)
        
        # Setup
        self._tempdir = os.path.join(TEMP_BASEDIR, '%s-%i' % (type(self).__name__, os.getpid()))
        if not os.path.exists(self._tempdir):
            os.mkdir(self._tempdir)
            
        # Cleanup
        atexit.register(shutil.rmtree, self._tempdir)
        
    def start(self, station, chan0, navg, nchan, chan_bw, npol, pols):
        """
        Set the metadata for the measurement sets and create the template.
        """
        
        # Setup
        tint = navg / CHAN_BW
        time_step = navg * (int(FS) / int(CHAN_BW))
        freq = numpy.arange(nchan)*chan_bw + chan_to_freq(chan0)
        if not isinstance(pols, (tuple, list)):
            pols = [p.strip().rstrip() for p in pols.split(',')]
            
        # Create the template
        self._template = os.path.join(self._tempdir, 'template')
        create_ms(self._template, station, tint, freq, pols)
        
        # Save
        self._station = station
        self._tint = tint
        self._time_step = time_step
        self._nant = len(self._station.antennas)
        self._freq = freq
        self._nchan = nchan
        self._pols = [STOKES_CODES[p] for p in pols]
        self._npol = len(self._pols)
        self._nbl = self._nant*(self._nant + 1) // 2
        self._started = True
        
    def write(self, time_tag, data):
        dt = timetag_to_datetime(time_tag)
        tstart = timetag_to_astropy(time_tag)
        tstop  = timetag_to_astropy(time_tag + self._time_step // 2)
        tcent  = timetag_to_astropy(time_tag + self._time_step)
        
        # Make a copy of the template
        tagname = "%.0fMHz_%s" % (self._freq[0]/1e6, dt.strftime('%Y%m%d_%H%M%S'))
        tempname = os.path.join(self._tempdir, tagname)
        with open('/dev/null', 'wb') as dn:
            subprocess.check_call(['cp', '-r', self._template, tempname],
                                  stderr=dn)
            
        # Find the point overhead
        zen = [ct.sidereal_time('mean', self._station.lon).to_value('rad'),
               self._station.lat]
        
        # Update the time
        update_time(tempname, tstart, tcent, tstop)
        
        # Update the pointing direction
        update_pointing(tempname, *zen)
        
        # Fill in the main table
        update_data(tempname, data[0,...])
        
        # Save it to its final location
        filename = "%s_%s.tar.gz" % (self.filename, tagname)
        with open('/dev/null', 'wb') as dn:
            subprocess.check_call(['tar', 'czf', filename, tempname],
                                  stderr=dn, cwd=self._tempdir)
        shutil.rmtree(tempname)
        
    def stop(self):
        """
        Close out the file and then call the 'post_stop_task' method.
        """
        
        shutil.rmtree(self._template)
        try:
            os.rmdir(self._tempdir)
        except OSError:
            pass
            
        try:
            self.post_stop_task()
        except NotImplementedError:
            pass
