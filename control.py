import os
import json
import shutil
import threading
from datetime import datetime, timedelta

from common import LWATime
from reductions import *
from filewriter import HDF5Writer, MeasurementSetWriter

__all__ = ['BeamCommandProcessor', 'VisibilityCommandProcessor']


class CommandBase(object):
    """
    Base class to hold a data recording pipeline command.  It seems like a good
    idea right now.
    """
    
    _required = ('id',)
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
        name = kls.command_name.replace('HDF5', '').replace('MS', '')
        setattr(processor, name.lower(), kls)
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
        
        self.log.debug("%s - %s", self.command_name, *args)
        
    def log_info(self, *args):
        """
        Print an INFO line to the log with the command name prepended.
        """
        
        self.log.info("%s - %s", self.command_name, *args)
        
    def log_warning(self, *args):
        """
        Print a WARNING line to the log with the command name prepended.
        """
        
        self.log.warning("%s - %s", self.command_name, *args)
        
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
        
        self.log.fatal("%s - %s", self.command_name, *args)
        
    def action(self, *args, **kwds):
        """
        Action to be called when the command is processed.  It should accept
        only arguments in the same order as self._required.  It should also
        return a boolean of whether or not the command succeeded.
        """
        
        raise NotImplementedError("Must be overridden by the subclass.")
        
    def __call__(self, data):
        """
        Execute the command.  This first takes in the JSON-encoded string, 
        unpacks it to a dictionary, validates that all of the requied keywords
        are present, and then passes them to the self.action() method.
        """
        
        # Try to unpack if it isn't already a dictionary
        if not isinstance(data, dict):
            try:
                data = json.loads(data)
            except (TypeError, ValueError) as e:
                self.log_error("Failed to parse JSON: %s", str(e))
                
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
            return False
            
        # Action and return
        return self.action(*args, **kwds)


class HDF5Record(CommandBase):
    """
    Command to schedule a recording of a beam to an HDF5 file.  The input data 
    should have:
     * id - a MCS command id
     * start_mjd - the MJD for the start of the recording
     * start_mpm - the MPM for the start of the recording
     * duration_ms - the duration of the recording in ms
    """
    
    _required = ('id', 'start_mjd', 'start_mpm', 'duration_ms')
    _optional = ('stokes_mode', 'time_avg', 'chan_avg')
    
    def action(self, id, start_mjd, start_mpm, duration_ms, stokes_mode=None, time_avg=1, chan_avg=1):
        try:
            filename = os.path.join(self.directory, '%06i_%09i' % (start_mjd, id))
            start = LWATime(start_mjd, start_mpm/1000.0/86400.0, format='mjd', scale='utc').datetime
            duration = timedelta(seconds=duration_ms//1000, microseconds=duration_ms*1000 % 1000000)
            stop = start + duration
        except (TypeError, ValueError) as e:
            self.log_error("Failed to unpack command data: %s", str(e))
            return False
            
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
            return False
            
        op = self.filewriter_base(filename, start, stop, reduction=reduction_op, **self.filewriter_kwds)
        try:
            self.queue.append(op)
        except (TypeError, RuntimeError) as e:
            self.log_error("Failed to schedule recording: %s", str(e))
            return False
            
        return True


class MSRecord(CommandBase):
    """
    Command to schedule a recording of visibility data to a measurement set.
    The input data should have:
     * id - a MCS command id
     * start_mjd - the MJD for the start of the recording
     * start_mpm - the MPM for the start of the recording
     * duration_ms - the duration of the recording in ms
    """
    
    _required = ('id', 'start_mjd', 'start_mpm', 'duration_ms')
    
    def action(self, id, start_mjd, start_mpm, duration_ms):
        try:
            filename = os.path.join(self.directory, '%06i_%09i' % (start_mjd, id))
            start = LWATime(start_mjd, start_mpm/1000.0/86400.0, format='mjd', scale='utc').datetime
            duration = timedelta(seconds=duration_ms//1000, microseconds=duration_ms*1000 % 1000000)
            stop = start + duration
        except (TypeError, ValueError) as e:
            self.log_error("Failed to unpack command data: %s", str(e))
            return False
            
        op = self.filewriter_base(filename, start, stop, **self.filewriter_kwds)
        try:
            self.queue.append(op)
        except (TypeError, RuntimeError) as e:
            self.log_error("Failed to schedule recording: %s", str(e))
            return False
            
        return True


class Cancel(CommandBase):
    """
    Cancel a previously scheduled recording.  The input data should have:
     * id - a MCS command id
     * queue_number - scheduled recording queue number of cancel
    """
    
    _required = ('id', 'queue_number')
    
    def action(self, id, queue_number):
        try:
            self.queue[queue_number].cancel()
        except IndexError as e:
            self.log_error("Failed to cancel recording: %s", str(e))
            return False
            
        return True


class Delete(CommandBase):
    """
    Delete a file from the recording directory.  The input data should have:
     * id - a MCS command id
     * file_number - scheduled recording queue number of cancel
    """
    
    _required = ('id', 'file_number')
    
    def action(self, id, file_number):
        if self.queue.active is not None:
            self.log_error("Cannot delete while recording is active")
            return False
            
        try:
            filenames = glob.glob(os.path.join(self.directory, '*'))
            filenames.sort(key=lambda x: os.path.getmtime(x))
            if os.path.isdir(filenames[file_number]):
                shutil.rmtree(filenames[file_number])
            else:
                os.unlink(filenames[file_number])
        except (IndexError, OSError) as e:
            self.log_error("Failed to delete file: %s", str(e))
            return False
            
        return True


class CommandProcessorBase(object):
    """
    Base class for a command processor.
    """
    
    _commands = ()
    
    def __init__(self, log, directory, queue, filewriter_base, filewriter_kwds=None, shutdown_event=None):
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
        
        for cls in self._commands:
            cls.attach_to_processor(self)
            
    def main(self):
        pass


class BeamCommandProcessor(CommandProcessorBase):
    """
    Command processor for power beam data.  Supports:
     * record
     * cancel
     * delete
    """
    
    _commands = (HDF5Record, Cancel, Delete)
    
    def __init__(self, log, directory, queue, shutdown_event=None):
        CommandProcessorBase.__init__(self, log, directory, queue, HDF5Writer,
                                      shutdown_event=shutdown_event)


class VisibilityCommandProcessor(CommandProcessorBase):
    """
    Command processor for visibilitye data.  Supports:
     * record
     * cancel
     * delete
    """
    
    _commands = (MSRecord, Cancel, Delete)
    
    def __init__(self, log, directory, queue, is_tarred=False, shutdown_event=None):
        CommandProcessorBase.__init__(self, log, directory, queue, MeasurementSetWriter,
                                      filewriter_kwds={'is_tarred': is_tarred},
                                      shutdown_event=shutdown_event)
