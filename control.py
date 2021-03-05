import os
import json
import threading
from datetime import datetime, timedelta

from common import LWATime

__all__ = ['CommandProcessor',]


class CommandProcessor(object):
    def __init__(self, log, directory, queue, filewriter_base, filewriter_kwds={}, shutdown_event=None):
        self.log = log
        self.directory = directory
        self.queue = queue
        self.filewriter_base = filewriter_base
        self.filewriter_kwds = filewriter_kwds
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        
    def main(self):
        pass
        
    def record(self, data):
        try:
            data = json.loads(data)
        except ValueError as e:
            self.log.error("Failed to parse JSON: %s", str(e))
            return False
            
        try:
            id = data['id']
            mjd_start = data['mjd_start']
            mpm_start = data['mpm_start']
            duration_ms = data['duration_ms']
            
            filename = os.path.join(self.directory, '%06i_%09i' % (mjd_start, id))
            start = LWATime(mjd_start, mpm_start/1000.0/86400.0, format='mjd', scale='utc').datetime
            duration = timedelta(seconds=duration_ms//1000, microseconds=duration_ms*1000 % 1000000)
            stop = start + duration
        except KeyError as e:
            self.log.error("Failed to unpack command data: %s", str(e))
            return False
            
        op = self.filewriter_base(filename, start, stop, **self.filewriter_kwds)
        try:
            self.queue.append(op)
        except (TypeError, RuntimeError) as e:
            self.log.error("Failed to schedule recording: %s", str(e))
            return False
            
        return True
        
    def cancel(self, data):
        try:
            data = json.loads(data)
        except ValueError as e:
            self.log.error("Failed to parse JSON: %s", str(e))
            return False
            
        try:
            id = data['id']
            queuenumber = data['queue_number']
        except KeyError as e:
            self.log.error("Failed to unpack command data: %s", str(e))
            return False
            
        try:
            self.queue[idx].cancel()
        except IndexError as e:
            self.log.error("Failed to cancel recording: %s", str(e))
            return False
            
        return True
        
    def delete(self, data):
        try:
            data = json.loads(data)
        except ValueError as e:
            self.log.error("Failed to parse JSON: %s", str(e))
            return False
            
        try:
            id = data['id']
            filenumber = data['file_number']
        except KeyError as e:
            self.log.error("Failed to unpack command data: %s", str(e))
            return False
            
        try:
            filenames = glob.glob(os.path.join(self.directory, '*'))
            filenames.sort(key=lambda x: os.path.getmtime(x))
            os.unlink(filenames[filenumber])
        except (IndexError, OSError) as e:
            self.log.error("Failed to delete file: %s", str(e))
            return False
            
        return True
        
