import weakref
from bisect import bisect
from datetime import datetime, timedelta
from textwrap import fill as tw_fill

from filewriter import FileWriterBase

__all__ = ['OperationsQueue',]


class OperationsQueue(object):
    """
    Class to queue file writing operations.
    """
    
    def __init__(self):
        self._queue = []
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
        
    def append(self, fileop):
        """
        Add a new sub-class of FileWriterBase to the queue.  In the process,
        check for conflicts with existing queue entries and purge the queue of
        anything that has expired.
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
            del self._queue[self._queue.index(expiredop)]
            
        # Link it with the queue
        fileop._queue = weakref.proxy(self)
        
        # Put it in the right place
        idx = bisect([queueop.start_time for queueop in self._queue], fileop.start_time)
        self._queue.insert(idx, fileop)
        
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
            del self._queue[self._queue.index(expiredop)]
            
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
