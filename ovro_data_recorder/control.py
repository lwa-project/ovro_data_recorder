import os
import json
import shutil
import threading
from datetime import datetime, timedelta

from astropy.time import TimeDelta

from mnc.common import CHAN_BW, CLOCK, LWATime, synchronize_time
from mnc.mcs import MonitorPoint, CommandCallbackBase, Client

from ovro_data_recorder.reductions import *
from ovro_data_recorder.filewriter import DRXWriter, VoltageBeamWriter, HDF5Writer, MeasurementSetWriter

__all__ = ['PowerBeamCommandProcessor', 'VisibilityCommandProcessor',
           'VoltageBeamCommandProcessor', 'RawVoltageBeamCommandProcessor']


class CommandBase(object):
    """
    Base class to hold a data recording pipeline command.  It seems like a good
    idea right now.
    """
    
    _required = ('sequence_id',)
    _optional = ()
    
    def __init__(self, log, queue, directory, filewriter_base, filewriter_kwds=None):
        self.log = log
        self.queue = queue
        self.directory = directory
        self.filewriter_base = filewriter_base
        if filewriter_kwds is None:
            filewriter_kwds = {}
        self.filewriter_kwds = filewriter_kwds
        
    @classmethod
    def attach_to_processor(cls, processor):
        kls = cls(processor.log, processor.queue, processor.directory,
                  processor.filewriter_base, processor.filewriter_kwds)
        name = kls.command_name.replace('HDF5', '').replace('MS', '').replace('DRXRe', 'Re').replace('Raw', '')
        setattr(processor, name.lower(), kls)
        callback = CommandCallbackBase(processor.client.client)
        def wrapper(**kwargs):
            return kls(**kwargs)
        callback.action = wrapper
        processor.client.set_command_callback(name.lower(), callback)
        return kls
        
    @property
    def command_name(self):
        """
        Command (class) name.
        """
        
        return type(self).__name__
        
    def log_debug(self, *args):
        """
        Print a DEBUG line to the log with the command name prepended.
        """
        
        msg = "%s - "+args[0]
        self.log.debug(msg, self.command_name, *args[1:])
        
    def log_info(self, *args):
        """
        Print an INFO line to the log with the command name prepended.
        """
        
        msg = "%s - "+args[0]
        self.log.info(msg, self.command_name, *args[1:])
        
    def log_warning(self, *args):
        """
        Print a WARNING line to the log with the command name prepended.
        """
        
        msg = "%s - "+args[0]
        self.log.warning(msg, self.command_name, *args[1:])
        
    def log_error(self, *args):
        """
        Print an ERROR line to the log with the command name prepended.
        """
        
        msg = "%s - "+args[0]
        self.log.error(msg, self.command_name, *args[1:])
        
    def log_fatal(self, *args):
        """
        Print a FATAL line to the log with the command name prepended.
        """
        
        msg = "%s - "+args[0]
        self.log.fatal(msg, self.command_name, *args[1:])
        
    def action(self, *args, **kwds):
        """
        Action to be called when the command is processed.  It should accept
        only arguments in the same order as self._required.  It should also
        two-element tuple of (boolean of whether or not the command succeeded, 
        an info message or an empty dictionary).
        """
        
        raise NotImplementedError("Must be overridden by the subclass.")
        
    def __call__(self, **kwargs):
        """
        Execute the command.  This first takes in the JSON-encoded string, 
        unpacks it to a dictionary, validates that all of the requied keywords
        are present, and then passes them to the self.action() method.
        """
        
        # Gather it up
        data = kwargs
        
        # Build up the argments requested of die trying
        try:
            args = [data[key] for key in self._required]
            kwds = {}
            for key in self._optional:
                try:
                    kwds[key] = data[key]
                except KeyError:
                    pass
        except KeyError:
            missing = [key for key in self._required if key not in data]
            self.log_error("Missing required keywords - %s", ' '.join(missing))
            return False, "Missing required keywords - %s" % (' '.join(missing),)
            
        # Action and return
        return self.action(*args, **kwds)


class Ping(CommandBase):
    """
    Command to simply reply to.  The input data should have:
     * id - a MCS command id
    """
    
    _required = ('sequence_id',)
    
    def action(self, sequence_id):
        return True, 'pong'


class Sync(CommandBase):
    """
    Command to force a time sync on the subsystem using `ntddate`.  The input data
    should have:
     * id - a MCS command id
     * server - the NTP server to use
    """
    
    _required = ('sequence_id', 'server')
    
    def action(self, sequence_id, server):
        status = synchronize_time(server)
        if not status:
            self.log_error("Failed to set time against %s", server)
        return status, str(status)


class HDF5Record(CommandBase):
    """
    Command to schedule a recording of a beam to an HDF5 file.  The input data 
    should have:
     * id - a MCS command id
     * start_mjd - the MJD for the start of the recording or "now" to start the
       recording 15 s from when the command is received
     * start_mpm - the MPM for the start of the recording
     * duration_ms - the duration of the recording in ms
    """
    
    _required = ('sequence_id', 'start_mjd', 'start_mpm', 'duration_ms')
    _optional = ('stokes_mode', 'time_avg', 'chan_avg')
    
    def action(self, sequence_id, start_mjd, start_mpm, duration_ms, stokes_mode=None, time_avg=1, chan_avg=1):
        try:
            if start_mjd == "now":
                start = LWATime.now() + TimeDelta(15, format='sec')
                start_mjd = int(start.mjd)
                start = start.datetime
            else:
                start = LWATime(start_mjd, start_mpm/1000.0/86400.0, format='mjd', scale='utc').datetime
            filename = os.path.join(self.directory, '%06i_%12s%7s' % (start_mjd,
                                                                      start.strftime('%H%M%S%f'),
                                                                      sequence_id[:7]))
            duration = timedelta(seconds=duration_ms//1000, microseconds=duration_ms*1000 % 1000000)
            stop = start + duration
            if time_avg not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
                raise AssertionError(f"invalid value for 'time_avg': {time_avg}")
            if chan_avg not in (1, 2, 4, 8, 16, 32, 64):
                raise AssertionError(f"invalid value for 'chan_avg': {chan_avg}")
        except (TypeError, ValueError, AssertionError) as e:
            self.log_error("Failed to unpack command data: %s", str(e))
            return False, "Failed to unpack command data: %s" % str(e)
            
        if stokes_mode in (None, 'XXYYCRCI'):
            reduction_op = XXYYCRCI(time_avg=time_avg, chan_avg=chan_avg)
        elif stokes_mode == 'XXYY':
            reduction_op = XXYY(time_avg=time_avg, chan_avg=chan_avg)
        elif stokes_mode == 'CRCI':
            reduction_op = CRCI(time_avg=time_avg, chan_avg=chan_avg)
        elif stokes_mode == 'IQUV':
            reduction_op = IQUV(time_avg=time_avg, chan_avg=chan_avg)
        elif stokes_mode == 'IV':
            reduction_op = IV(time_avg=time_avg, chan_avg=chan_avg)
        else:
            self.log_error("Unknown Stokes mode: %s", stokes_mode)
            return False, "Unknown Stokes mode: %s" % stokes_mode
            
        op = self.filewriter_base(filename, start, stop, reduction=reduction_op, **self.filewriter_kwds)
        try:
            self.queue.append(op)
        except (TypeError, RuntimeError) as e:
            self.log_error("Failed to schedule recording: %s", str(e))
            return False, "Failed to schedule recording: %s" % str(e)
            
        self.log_info("Scheduled recording for %s to %s to %s", start, stop, filename)
        return True, {'filename': filename}


class MSStart(CommandBase):
    """
    Command to schedule a recording start of visibility data to a measurement set.
    The input data should have:
     * id - a MCS command id
     * mjd - the MJD for the start of the recording or "now" to start the
       recording 15 s from when the command is received
     * mpm - the MPM for the start of the recording
    """
    
    _required = ('sequence_id', 'mjd', 'mpm')
    
    def action(self, sequence_id, mjd, mpm):
        try:
            filename = os.path.join(self.directory, '')
            if mjd == "now":
                start = LWATime.now() + TimeDelta(15, format='sec')
                start = start.datetime
            else:
                start = LWATime(mjd, mpm/1000.0/86400.0, format='mjd', scale='utc').datetime
            duration = timedelta(days=365)
            stop = start + duration
        except (TypeError, ValueError) as e:
            self.log_error("Failed to unpack command data: %s", str(e))
            return False, "Failed to unpack command data: %s" % str(e)
            
        op = self.filewriter_base(filename, start, stop, **self.filewriter_kwds)
        try:
            self.queue.append(op)
        except (TypeError, RuntimeError) as e:
            self.log_error("Failed to schedule recording start: %s", str(e))
            return False, "Failed to schedule recording start: %s" % str(e)
            
        self.log_info("Scheduled recording start for %s to %s to %s", start, stop, filename)
        return True, {'filename': filename}


class DRXRecord(CommandBase):
    """
    Command to schedule a recording of a voltage beam to a DRX file.  The
    input data should have:
     * id - a MCS command id
     * start_mjd - the MJD for the start of the recording or "now" to start the
       recording 15 s from when the command is received
     * start_mpm - the MPM for the start of the recording
     * duration_ms - the duration of the recording in ms
    """
    
    _required = ('sequence_id', 'beam', 'start_mjd', 'start_mpm', 'duration_ms')
    
    def action(self, sequence_id, beam, start_mjd, start_mpm, duration_ms):
        try:
            if start_mjd == "now":
                start = LWATime.now() + TimeDelta(15, format='sec')
                start_mjd = int(start.mjd)
                start = start.datetime
            else:
                start = LWATime(start_mjd, start_mpm/1000.0/86400.0, format='mjd', scale='utc').datetime
            filename = os.path.join(self.directory, '%06i_%12s%7s' % (start_mjd,
                                                                      start.strftime('%H%M%S%f'),
                                                                      sequence_id[:7]))
            duration = timedelta(seconds=duration_ms//1000, microseconds=duration_ms*1000 % 1000000)
            stop = start + duration
        except (TypeError, ValueError) as e:
            self.log_error("Failed to unpack command data: %s", str(e))
            return False, "Failed to unpack command data: %s" % str(e)
            
        op = self.filewriter_base(filename, beam, start, stop, **self.filewriter_kwds)
        try:
            self.queue.append(op)
        except (TypeError, RuntimeError) as e:
            self.log_error("Failed to schedule recording: %s", str(e))
            return False, "Failed to schedule recording start: %s" % str(e)
            
        self.log_info("Scheduled recording for %s to %s to %s", start, stop, filename)
        return True, {'filename': filename}


class RawRecord(CommandBase):
    """
    Command to schedule a recording of a voltage beam to a raw voltage beam file.  The
    input data should have:
     * id - a MCS command id
     * start_mjd - the MJD for the start of the recording or "now" to start the
       recording 15 s from when the command is received
     * start_mpm - the MPM for the start of the recording
     * duration_ms - the duration of the recording in ms
    """
    
    _required = ('sequence_id', 'beam', 'start_mjd', 'start_mpm', 'duration_ms')
    
    def action(self, sequence_id, beam, start_mjd, start_mpm, duration_ms):
        try:
            if start_mjd == "now":
                start = LWATime.now() + TimeDelta(15, format='sec')
                start_mjd = int(start.mjd)
                start = start.datetime
            else:
                start = LWATime(start_mjd, start_mpm/1000.0/86400.0, format='mjd', scale='utc').datetime
            filename = os.path.join(self.directory, '%06i_%12s%7s' % (start_mjd,
                                                                      start.strftime('%H%M%S%f'),
                                                                      sequence_id[:7]))
            duration = timedelta(seconds=duration_ms//1000, microseconds=duration_ms*1000 % 1000000)
            stop = start + duration
        except (TypeError, ValueError) as e:
            self.log_error("Failed to unpack command data: %s", str(e))
            return False, "Failed to unpack command data: %s" % str(e)
            
        op = self.filewriter_base(filename, beam, start, stop, **self.filewriter_kwds)
        try:
            self.queue.append(op)
        except (TypeError, RuntimeError) as e:
            self.log_error("Failed to schedule recording: %s", str(e))
            return False, "Failed to schedule recording start: %s" % str(e)
            
        self.log_info("Scheduled recording for %s to %s to %s", start, stop, filename)
        return True, {'filename': filename}


class Cancel(CommandBase):
    """
    Cancel a previously scheduled recording.  The input data should have:
     * id - a MCS command id
     * queue_number - scheduled recording queue number of cancel
    """
    
    _required = ('sequence_id',)
    _optional = ('queue_number', 'filename',)
    
    def action(self, sequence_id, queue_number=None, filename=None):
        if queue_number is None and filename is None:
            self.log_error("Must specify a queue number or filename")
            return False, "Must specify a queue number or filename"
            
        if filename is not None:
            op = self.queue.find_entry_by_filename(filename)
        else:
            try:
                op = self.queue[queue_number]
            except IndexError as e:
                self.log_error("Invalid queue entry number")
                return False, "Invalid queue entry number"
                
        try:
            filename = op.filename
            start = op.start_time
            stop = op.stop_time
            op.cancel()
            self.queue.clean()
        except Exception as e:
            self.log_error("Failed to cancel recording: %s", str(e))
            return False, "Failed to cancel recording: %s" % str(e)
            
        self.log_info("Canceled recording for %s to %s to %s", start, stop, filename)
        return True, {'filename': filename}


class MSStop(CommandBase):
    """
    Command to schedule a recording stop of visibility data to a measurement set.
    The input data should have:
     * id - a MCS command id
     * mjd - the MJD for the start of the recording or "now" to stop the
       recording 15 s from when the command is received
     * mpm - the MPM for the start of the recording
    """
    
    _required = ('sequence_id', 'mjd', 'mpm')
    
    def action(self, sequence_id, mjd, mpm):
        try:
            if mjd == "now":
                stop = LWATime.now() + TimeDelta(15, format='sec')
                stop = stop.datetime
            else:
                stop = LWATime(mjd, mpm/1000.0/86400.0, format='mjd', scale='utc').datetime
        except (TypeError, ValueError) as e:
            self.log_error("Failed to unpack command data: %s", str(e))
            return False, "Failed to unpack command data: %s" % str(e)
            
        try:
            op = self.queue.find_entry_active_at_datetime(stop)
            if op is not None:
                filename = op.filename
                start = op.start_time
                if op.is_started:
                    op.cancel()
                else:
                    op.stop_time = stop
                    op._padded_stop_time = op.stop_time + op._margin
                self.queue.clean()
            else:
                self.log_error("Failed to find operation active at the specified stop time")
                return False, "Failed to find operation active at the specified stop time"
        except Exception as e:
            self.log_error("Failed to stop recording: %s", str(e))
            return False, "Failed to stop recording: %s" % str(e)
            
        self.log_info("Stopped recording started at %s to %s", start, filename)
        return True, {'filename': filename}


class Delete(CommandBase):
    """
    Delete a file from the recording directory.  The input data should have:
     * id - a MCS command id
     * file_number - scheduled recording queue number of cancel
    """
    
    _required = ('sequence_id', 'file_number')
    
    def action(self, sequence_id, file_number):
        if self.queue.active is not None:
            self.log_error("Cannot delete while recording is active")
            return False, "Cannot delete while recording is active"
            
        try:
            filenames = glob.glob(os.path.join(self.directory, '*'))
            filenames.sort(key=lambda x: os.path.getmtime(x))
            filename = filenames[file_number]
            if os.path.isdir(filename):
                shutil.rmtree(filename)
            else:
                os.unlink(filename)
        except (IndexError, OSError) as e:
            self.log_error("Failed to delete file: %s", str(e))
            return False, "Failed to delete file: %s" % str(e)
            
        self.log_info("Deleted recording %s", filename)
        return True, {'filename': filename}


class DRX(CommandBase):
    """
    Command to the tuning/filter/gain for a voltage beam.  The input data should
    have:
     * id - a MCS command id
     * beam - beam number (1 or 2)
     * tuning - tuning number (1 or 2)
     * central_freq - central frequency in Hz
     * filter - filter code (0-7; 7 = 19.6 MHz)
     * gain - gain value (0-15)
    """
    
    _required = ('sequence_id', 'beam', 'tuning', 'central_freq', 'filter', 'gain')
    _optional = ('subslot')
    
    _bandwidths = {1:   250000, 
                   2:   500000, 
                   3:  1000000, 
                   4:  2000000, 
                   5:  4900000, 
                   6:  9800000, 
                   7: 19600000}
    
    def action(self, sequence_id, beam, tuning, central_freq, filter, gain, subslot=0):
        try:
            if beam not in (1,2):
                raise AssertionError(f"invalid value for 'beam': {beam}")
            if tuning not in (1,2):
                raise AssertionError(f"invalid value for 'tuning': {tuning}")
            if filter not in self._bandwidths.keys():
                raise AssertionError(f"invalid value for 'filter': {filter}")
            if (central_freq <= self._bandwidths[filter]/2) \
                   or (central_freq >= (CLOCK/2 - self._bandwidths[filter]/2)):
                raise AssertionError(f"invalid value for 'central_freq': {central_freq}")
            if gain < 0 or gain > 15:
                raise AssertionError(f"invalid value for 'gain': {gain}")
            if subslot < 0 or subslot > 99:
                raise AssertionError(f"invalid value for 'subslot': {subslot}")
        except (TypeError, AssertionError) as e:
            self.log_error("Failed to unpack command data: %s", str(e))
            return False, "Failed to unpack command data: %s" % str(e)
            
        # Put it in the queue
        self.queue.append(beam, tuning, central_freq, filter, gain)
        
        self.log_info("Beam %i, tuning %i to %.3f MHz at filter %i and gain %i", beam,
                                                                                 tuning,
                                                                                 central_freq/1e6,
                                                                                 filter,
                                                                                 gain)
        return True, "success"


class BND(CommandBase):
    """
    Command to the frequency downselection of a raw voltage beam.  The input data should
    have:
     * id - a MCS command id
     * beam - beam number (1 or 2)
     * central_freq - central frequency in Hz
     * bw - bandwidth in Hz
    """
    
    _required = ('sequence_id', 'beam', 'central_freq', 'bw')
    
    _min_bandwidth = 3 * CHAN_BW
    _max_bandwidth = 200 * CHAN_BW
    
    def action(self, sequence_id, beam, central_freq, bw):
        try:
            if beam not in (1,2):
                raise AssertionError(f"invalid value for 'beam': {beam}")
            if (central_freq <= bw/2) \
                   or (central_freq >= (CLOCK/2 - bw/2)):
                raise AssertionError(f"invalid value for 'central_freq': {central_freq}")
            if bw < _min_bandwidth or bw > _max_bandwidth:
                raise AssertionError(f"invalid value for 'bw': {bw}")
        except (TypeError, AssertionError) as e:
            self.log_error("Failed to unpack command data: %s", str(e))
            return False, "Failed to unpack command data: %s" % str(e)
            
        # Put it in the queue
        self.queue.append(beam, central_freq, bw)
        
        self.log_info("Raw Voltage Beam %i, to %.3f MHz with bandwidth %.3f MHz", beam,
                                                                              central_freq/1e6,
                                                                              bw/1e6)
        return True, "success"


class CommandProcessorBase(object):
    """
    Base class for a command processor.
    """
    
    _commands = ()
    
    def __init__(self, log, id, directory, queue, filewriter_base, filewriter_kwds=None, shutdown_event=None):
        self.log = log
        self.directory = directory
        self.queue = queue
        self.filewriter_base = filewriter_base
        if filewriter_kwds is None:
            filewriter_kwds = {}
        self.filewriter_kwds = filewriter_kwds
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        
        self.client = Client(id)
        
        for cls in self._commands:
            cls.attach_to_processor(self)
            
    def main(self):
        pass


class PowerBeamCommandProcessor(CommandProcessorBase):
    """
    Command processor for power beam data.  Supports:
     * ping
     * sync
     * record
     * cancel
     * delete
    """
    
    _commands = (Ping, Sync, HDF5Record, Cancel, Delete)
    
    def __init__(self, log, id, directory, queue, shutdown_event=None):
        CommandProcessorBase.__init__(self, log, id, directory, queue, HDF5Writer,
                                      shutdown_event=shutdown_event)


class VisibilityCommandProcessor(CommandProcessorBase):
    """
    Command processor for visibility data.  Supports:
     * ping
     * sync
     * start
     * stop
    """
    
    _commands = (Ping, Sync, MSStart, MSStop)
    
    def __init__(self, log, id, directory, queue, nint_per_file=1, is_tarred=False, shutdown_event=None):
        CommandProcessorBase.__init__(self, log, id, directory, queue, MeasurementSetWriter,
                                      filewriter_kwds={'nint_per_file': nint_per_file,
                                                       'is_tarred': is_tarred},
                                      shutdown_event=shutdown_event)


class VoltageBeamCommandProcessor(CommandProcessorBase):
    """
    Command processor for T-engine'd voltage beam data.  Supports:
     * ping
     * sync
     * record
     * cancel
     * delete
     * drx
    """
    
    _commands = (Ping, Sync, DRXRecord, Cancel, Delete, DRX)
    
    def __init__(self, log, id, directory, queue, drx_queue, shutdown_event=None):
        CommandProcessorBase.__init__(self, log, id, directory, queue, DRXWriter,
                                      shutdown_event=shutdown_event)
        self.drx.queue = drx_queue


class RawVoltageBeamCommandProcessor(CommandProcessorBase):
    """
    Command processor for raw voltage beam data.  Supports:
     * ping
     * sync
     * record
     * cancel
     * delete
     * drx
    """
    
    _commands = (Ping, Sync, RawRecord, Cancel, Delete, BND)
    
    def __init__(self, log, id, directory, queue, bnd_queue, shutdown_event=None):
        CommandProcessorBase.__init__(self, log, id, directory, queue, VoltageBeamWriter,
                                      shutdown_event=shutdown_event)
        self.bnd.queue = bnd_queue
