import os
import sys
import h5py
import json
import time
import numpy
import atexit
import shutil
import subprocess
from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
from textwrap import fill as tw_fill

from mnc.common import *

from lwahdf import *
from lwams import *


__all__ = ['FileWriterBase', 'DRXWriter', 'HDF5Writer', 'MeasurementSetWriter']


# Temporary file directory
_TEMP_BASEDIR = "/fast/pipeline/temp/"


class FileWriterBase(object):
    """
    Class to represent a file to write data to for the specified time period.
    """
    
    # +/- time margin for whether or not a file is active or finished.
    _margin = timedelta(seconds=1)
    
    def __init__(self, filename, start_time, stop_time, reduction=None):
        self.filename = os.path.abspath(filename)
        self.start_time = start_time
        self.stop_time = stop_time
        self.reduction = reduction
        
        self._padded_start_time  = self.start_time - self._margin
        self._padded_stop_time = self.stop_time + self._margin
        
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
        """
        Pipeline-lag aware version of `datetime.datetime.utcnow` when the file
        has been linked with an `operations.OperationQueue`.  Otherwise, it is
        the same as `datetime.datetime.utcnow`.
        """
        
        now = datetime.utcnow()
        if self._queue is not None:
            now = now - self._queue.lag
        return now
        
    @property
    def is_active(self):
        """
        Whether or not the file should be considered active, i.e., the current
        time is within its scheduled window.
        """
        
        now = self.utcnow()
        return (now >= self._padded_start_time) and (now <= self._padded_stop_time)
        
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
        return now > self._padded_stop_time
        
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
        else:
            self.start_time = datetime.utcnow() - 2*self._margin
            self._padded_start_time  = self.start_time - self._margin
        self.stop_time = datetime.utcnow() - 4*self._margin
        self._padded_stop_time = self.stop_time + self._margin


class DRXWriter(FileWriterBase):
    """
    Sub-class of :py:class:`FileWriterBase` that writes data to a raw DRX file.
    """
    
    def __init__(self, filename, beam, start_time, stop_time):
        FileWriterBase.__init__(self, filename, start_time, stop_time)
        self.beam = beam
        
    def start(self):
        """
        Start the file writer and return the open file handle.
        """
        
        self._interface = open(self.filename, 'wb')
        self._started = True
        
        return self._interface


class HDF5Writer(FileWriterBase):
    """
    Sub-class of :py:class:`FileWriterBase` that writes data to a HDF5 file.
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
        chunks = int((self.stop_time - self.start_time).total_seconds() / (navg / CHAN_BW)) + 1
        
        # Create and fill
        self._interface = create_hdf5(self.filename, beam)
        self._freq = set_frequencies(self._interface, freq)
        self._time = set_time(self._interface, navg / CHAN_BW, chunks)
        self._time_step = navg * int(round(FS/CHAN_BW))
        self._start_time_tag = LWATime(self.start_time, format='datetime', scale='utc').tuple
        self._stop_time_tag = LWATime(self.stop_time, format='datetime', scale='utc').tuple
        self._pols = set_polarization_products(self._interface, pols, chunks)
        self._counter = 0
        self._counter_max = chunks
        self._started = True

        # Enable concurrent access to the file
        self._interface.swmr_mode = True
        self._freq.flush()
        self._last_flush = time.time()
        
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
            
        # Timestamps
        time_tags = [LWATime(time_tag+i*self._time_step, format='timetag').tuple for i in range(data.shape[0])]
        
        # Data selection
        if time_tags[0] < self._start_time_tag:
            ## Lead in
            offset = bisect_left(time_tags, self._start_time_tag)
            size = min([self._counter_max - self._counter, len(time_tags) - offset])
            range_start = offset
        elif time_tags[-1] > self._stop_time_tag:
            ## Flush out
            offset = bisect_right(time_tags, self._stop_time_tag)
            size = min([self._counter_max - self._counter, offset])
            range_start = 0
        else:
            ## Fully contained
            size = min([self._counter_max - self._counter, len(time_tags)])
            range_start = 0
            
        try:
            # Write
            ## Timestamps
            self._time[self._counter:self._counter+size] = time_tags[range_start:range_start+size]
            ## Data
            for i in range(data.shape[-1]):
                self._pols[i][self._counter:self._counter+size,:] = data[range_start:range_start+size,0,:,i]
            # Update the counter
            self._counter += size
            # Flush every 10 s
            if time.time() - self._last_flush > 10:
                self._time.flush()
                for i in range(data.shape[-1]):
                    self._pols[i].flush()
                self._last_flush = time.time()
                
        except ValueError:
            # If we are here that probably means the file has been closed
            pass


class MeasurementSetWriter(FileWriterBase):
    """
    Sub-class of :py:class:`FileWriterBase` that writes data to a measurement
    set.  Each call to write leads to a new measurement set.
    """
    
    def __init__(self, filename, start_time, stop_time, nint_per_file=1, is_tarred=True):
        FileWriterBase.__init__(self, filename, start_time, stop_time, reduction=None)
        
        # Setup
        self._tempdir = os.path.join(_TEMP_BASEDIR, '%s-%i' % (type(self).__name__, os.getpid()))
        if not os.path.exists(self._tempdir):
            os.mkdir(self._tempdir)
        self.nint_per_file = nint_per_file
        self.is_tarred = is_tarred
        
        # Cleanup
        atexit.register(shutil.rmtree, self._tempdir, ignore_errors=True)
        
    def start(self, station, chan0, navg, nchan, chan_bw, npol, pols):
        """
        Set the metadata for the measurement sets and create the template.
        """
        
        # Setup
        tint = navg / FS
        time_step = navg
        freq = numpy.arange(nchan)*chan_bw + chan_to_freq(chan0)
        if not isinstance(pols, (tuple, list)):
            pols = [p.strip().rstrip() for p in pols.split(',')]
            
        # Refresh the station - needed for fast visibilities
        try:
            station.refresh()
        except AttributeError:
            pass
            
        # Create the template
        self._template = os.path.join(self._tempdir, 'template')
        create_ms(self._template, station, tint, freq, pols, nint=self.nint_per_file)
        
        # Update the file completion margin
        self._margin = timedelta(seconds=max([1, int(round(time_step / FS))]))
        self._padded_stop_time = self.stop_time + self._margin
        
        # Save
        self._station = station
        self._tint = tint
        self._time_step = time_step
        self._nant = len(self._station.antennas)
        self._freq = freq
        self._nchan = nchan
        self._raw_pols = pols
        self._pols = [STOKES_CODES[p] for p in pols]
        self._npol = len(self._pols)
        self._nint = self.nint_per_file
        self._nbl = self._nant*(self._nant + 1) // 2
        self._counter = 0
        self._started = True
        
    def write(self, time_tag, data):
        tstart = LWATime(time_tag, format='timetag', scale='utc')
        tcent  = LWATime(time_tag + self._time_step // 2, format='timetag', scale='utc')
        tstop  = LWATime(time_tag + self._time_step, format='timetag', scale='utc')
        
        # Build a template for the file
        if self._counter == 0:
            self.tagpath = os.path.join(self.filename,
                                        f"{self._freq[0]/1e6:.0f}MHz",
                                        tstart.datetime.strftime("%Y-%m-%d"),
                                        tstart.datetime.strftime("%H"))
            if not os.path.exists(self.tagpath):
                os.makedirs(self.tagpath, exist_ok=True)
            self.tagname = "%s_%.0fMHz.ms" % (tstart.datetime.strftime('%Y%m%d_%H%M%S'), self._freq[0]/1e6)
            self.tempname = os.path.join(self._tempdir, self.tagname)
            with open('/dev/null', 'wb') as dn:
                subprocess.check_call(['cp', '-r', self._template, self.tempname],
                                      stderr=dn)
                
        # Find the point overhead
        zen = get_zenith(self._station, tcent)
        
        # Update the time
        update_time(self.tempname, self._counter, tstart, tcent, tstop)
        
        # Update the pointing direction
        update_pointing(self.tempname, self._counter, *zen)
        
        # Fill in the main table
        update_data(self.tempname, self._counter, data[0,...])
        
        # Save it to its final location
        self._counter += 1
        if self._counter == self._nint:
            self.tagname = os.path.join(self.tagpath, self.tagname)
            if self.is_tarred:
                filename = os.path.join(self.filename, "%s.tar" % self.tagname)
                save_cmd = ['tar', 'cf', filename, os.path.basename(self.tempname)]
            else:
                filename = os.path.join(self.filename, self.tagname)
                save_cmd = ['cp', '-rf', self.tempname, filename]
            with open('/dev/null', 'wb') as dn:
                try:
                    subprocess.check_call(save_cmd, stderr=dn, cwd=self._tempdir)
                    shutil.rmtree(self.tempname, ignore_errors=True)
                    self._counter = 0
                except subprocess.CalledProcessError as e:
                    shutil.rmtree(self.tempname, ignore_errors=True)
                    self._counter = 0
                    raise e
                    
    def stop(self):
        """
        Close out the file and then call the 'post_stop_task' method.
        """
        
        try:
            shutil.rmtree(self._template, ignore_errors=True)
        except OSError:
            pass
        try:
            os.rmdir(self._tempdir)
        except OSError:
            pass
            
        try:
            self.post_stop_task()
        except NotImplementedError:
            pass
