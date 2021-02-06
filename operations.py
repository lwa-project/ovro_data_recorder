from bisect import bisect
from datetime import datetime
from collections import deque

from filewriter import FileWriterBase


class OperationsQueue(object):
    def __init__(self):
        self._queue = deque([])
        
    def __repr__(self):
        return "<%s at %x>" % (type(self).__name__, id(self))
        
    def __len__(self):
        return len(self._queue)
        
    @property
    def empty(self):
        return len(self) == 0
        
    def append(self, fileop):
        if not isinstance(fileop, FileWriterBase):
            raise TypeError("Expected a sub-class of FileWriterBase")
            
        # Conflict checking and cleaning
        to_remove = []
        for queueop in self._queue:
            if (fileop.start_time >= queueop.start_time) and (fileop.start_time <= queueop.stop_time):
                raise RuntimeError("Operation starts during a previously scheduled operation")
            if (fileop.stop_time >= queueop.stop_time) and (fileop.stop_time <= queueop.stop_time):
                raise RuntimeError("Operation continues into a previously scheduled operation")
            if (fileop.start_time <= queueop.start_time) and (fileop.stop_time >= queueop.stop_time):
                raise RuntimeError("Operation overlaps with a previously scheduled operation")
            if queueop.is_expired:
                to_remove.append(queueop)
        for expiredop in to_remove:
            del self._queue[self._queue.index(expiredop)]
            
        # Put it in the right place
        idx = bisect([queueop.start_time for queueop in self._queue])
        self._queue.insert(idx, fileop)
        
    def clean(self):
        to_remove = []
        for queueop in self._queue:
            if queueop.is_expired:
                to_remove.append(queueop)
        for expiredop in to_remove:
            del self._queue[self._queue.index(expiredop)]
            
    @property
    def active(self):
        activeop = None
        for queueop in self._queue:
            if queueop.is_active:
                activeop = queuop
                break
        return activeop
