import weakref
from bisect import bisect
from datetime import datetime, timedelta
from textwrap import fill as tw_fill
from collections import deque

from filewriter import FileWriterBase

__all__ = ['OperationsQueueBase', 'FileOperationsQueue', 'DrxOperationsQueue']


class OperationsQueueBase(object):
    """
    Base class for queuing operations.
    """
    
    def __init__(self):
        self._queue = deque([])
        self._last = None
        
        self._lag = timedelta()
        
    def __repr__(self):
        output = "<%s at 0x%x>" % (type(self).__name__, id(self))
        return tw_fill(output, subsequent_indent='    ')
        
    def __len__(self):
        return len(self._queue)
        
    def __getitem__(self, idx):
        return self._queue[idx]
        
    def update_lag(self, latest_dt):
        """
        Given a datetime instance that represents the last time processes by
        the pipeline, update the internal lag.
        """
        
        self._lag = datetime.utcnow() - latest_dt
        
    @property
    def lag(self):
        """
        The current pipeline lag as a timedelta instance.
        """
        
        return self._lag
        
    @property
    def empty(self):
        """
        Whether or not there are any operations in the queue.
        """
        
        return len(self) == 0
        
    def append(self, op):
        """
        Method to call when adding a new entry to the queue.  To
        be overridden by sub-classes.
        """
        
        raise NotImplementedError


class FileOperationsQueue(OperationsQueueBase):
    """
    Class to queue file writing operations.
    """
    
    def find_entry_by_filename(self, filename):
        """
        Find the queue entry associated with the specified filename.  Returns the
        operation if found, None otherwise.
        """
        
        by_filename = None
        for op in self._queue:
            if os.path.basename(op.filename) == filename:
                by_filename = op
                break
        return by_filename
        
    def find_entry_active_at_datetime(self, dt):
        """
        Find which queue entry would be active at the specified datetime instance.
        Returns the operation if found, None otherwise.
        """
        
        active_at = None
        for op in self._queue:
            if dt >= op.start_time and dt <= op.stop_time:
                active_at = op
                break
        return active_at
        
    def append(self, fileop):
        """
        Add a new sub-class of :py:class:`FileWriterBase` to the queue.  In the
        process, check for conflicts with existing queue entries and purge the
        queue of anything that has expired.
        """
        
        if not isinstance(fileop, FileWriterBase):
            raise TypeError("Expected a sub-class of FileWriterBase")
        if fileop._padded_start_time < datetime.utcnow() - timedelta(seconds=2):
            raise RuntimeError("Insufficient advanced notice %s" % (datetime.utcnow()-fileop.start_time,))
            
        # Conflict checking and cleaning
        to_remove = []
        for queueop in self._queue:
            if ((fileop._padded_start_time >= queueop._padded_start_time) \
                and (fileop._padded_start_time <= queueop._padded_stop_time)):
                raise RuntimeError("Operation starts during a previously scheduled operation")
            if ((fileop._padded_stop_time >= queueop._padded_stop_time) \
                and (fileop._padded_stop_time <= queueop._padded_stop_time)):
                raise RuntimeError("Operation continues into a previously scheduled operation")
            if ((fileop._padded_start_time <= queueop._padded_start_time) \
                and (fileop._padded_stop_time >= queueop._padded_stop_time)):
                raise RuntimeError("Operation overlaps with a previously scheduled operation")
            if queueop.is_expired:
                to_remove.append(queueop)
        for expiredop in to_remove:
            self._last = expiredop
            self._queue.remove(expiredop)
            
        # Link it with the queue
        fileop._queue = weakref.proxy(self)
        
        # Put it in the right place
        idx = bisect([queueop.start_time for queueop in self._queue], fileop.start_time)
        if idx == 0:
            self._queue.appendleft(fileop)
        elif idx == len(self._queue):
            self._queue.append(fileop)
        else:
            self._queue.append(None)
            self._queue[idx+1:] = self._queue[idx:-1]
            self._queue[idx] = fileop
            
    def clean(self):
        """
        Purge the queue of anything that has expired.
        """
        
        to_remove = []
        for queueop in self._queue:
            if queueop.is_expired:
                to_remove.append(queueop)
        for expiredop in to_remove:
            self._last = expiredop
            self._queue.remove(expiredop)
            
    @property
    def active(self):
        """
        The active file writer operation or None if there is not one.
        """
        
        activeop = None
        try:
            if self._queue[0].is_active:
                activeop = self._queue[0]
        except IndexError:
            pass
        return activeop
        
    @property
    def previous(self):
        """
        The last file writer operation or None if there is not one.
        """
        
        return self._last


class DrxOperationsQueue(OperationsQueueBase):
    """
    Class to queue changes to the voltage data streams.
    """
    
    def append(self, beam, tuning, central_freq, filter, gain):
        op = (beam, tuning, central_freq, filter, gain)
        self._queue.append(op)
        
    @property
    def active(self):
        """
        The active DRX command or None if there is not one.
        """
        
        activeop = None
        try:
            activeop = self._queue[0]
        except IndexError:
            pass
        return activeop
        
    def set_active_accepted(self):
        """
        Set the active command as accepted.
        """
        
        try:
            self._queue.popleft()
        except IndexError:
            return False
        return True
