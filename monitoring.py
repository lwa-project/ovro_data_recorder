import os
import sys
import glob
import time
import threading
from collections import deque

from bifrost.proclog import load_by_pid

__all__ = ['PerformanceLogger', 'StorageLogger', 'StatusLogger', 'GlobalLogger']


class PerformanceLogger(object):
    def __init__(self, log, queue=None, shutdown_event=None):
        self.log = log
        self.queue = queue
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        
        self._pid = os.getpid()
        self._state = deque([], 2)
        self._update()
        
    def _update(self):
        new_state = load_by_pid(self._pid)
        new_state_time = time.time()
        
        self._state.append((new_state_time,new_state))
        
    def main(self, once=False):
        while not self.shutdown_event.is_set():
            # Update the state
            self._update()
            
            # Get the pipeline lag, is possible
            lag = None
            if self.queue is not None:
                lag = self.queue.lag
                
            # Find the maximum acquire/process/reserve times
            acquire, process, reserve = 0.0, 0.0, 0.0
            for block,contents in self._state[1][1].items():
                try:
                    perf = contents['perf']
                except KeyError:
                    continue
                acquire = max([acquire, perf['acquire_time']])
                process = max([process, perf['process_time']])
                reserve = max([reserve, perf['reserve_time']])
                
            # Estimate the data rate and current missing data fracation
            rx_valid, rx_rate, missing_fraction = False, 0.0, 0.0
            good0, late0, missing0 = 0, 0, 0
            good1, late1, missing1 = 0, 0, 0
            for block,contents in self._state[0][1].items():
                if block[-8:] == '_capture':
                    rx_valid = True
                    good0 = contents['stats']['ngood_bytes']
                    late0 = contents['stats']['nlate_bytes']
                    missing0 = contents['stats']['nmissing_bytes']
            for block,contents in self._state[1][1].items():
                if block[-8:] == '_capture':
                    good1 = contents['stats']['ngood_bytes']
                    late1 = contents['stats']['nlate_bytes']
                    missing1 = contents['stats']['nmissing_bytes']
            try:
                rx_rate = (good1 - good1) / (self._state[1][0] - self._state[0][0])
                missing_fraction = (missing1 - missing0) / (good1 - good0 + missing1 - missing0)
            except (IndexError, ZeroDivisionError):
                rx_valid = False
                
            # Load average
            try:
                one, five, fifteen = os.getloadavg()
            except OSError:
                one, five, fifteen = None, None, None
                
            # Report
            self.log.debug("=== Performance Report ===")
            self.log.debug(" max acquire/process/reserve times: %.3f/%.3f/%.3f", acquire, process, reserve)
            if rx_valid:
                self.log.debug(" receive data rate: %.3f B/s", rx_rate)
                self.log.debug(" missing data fraction: %.3f%%", missing_fraction*100.0)
            if lag is not None:
                self.log.debug(" pipeline lag: %s", lag)
            if one is not None:
                self.log.debug(" load average: %.2f, %.2f, %.2f", one, five, fifteen)
            self.log.debug("===   ===")
            
            # Sleep
            if once:
                break
            time.sleep(10)


class StorageLogger(object):
    def __init__(self, log, directory, shutdown_event=None):
        self.log = log
        self.directory = directory
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        
        self._files = []
        
    def _update(self):
        self._files = glob.glob('./*')
        self._files.sort(key=lambda x: os.path.getmtime(x))
        
    def main(self, once=False):
        while not self.shutdown_event.is_set():
            # Update the state
            self._update()
            
            # Find the disk size and free space for the disk hosting the
            # directory - this should be quota-aware
            st = os.statvfs(self.directory)
            disk_free = st.f_bavail * st.f_frsize
            disk_total = st.f_blocks * st.f_frsize
            
            # Find the total size of all files
            total_size = sum([os.path.getsize(f) for f in self._files])
            
            # Find the most recent file
            try:
                latest = self._files[-1]
                latest_size = os.path.getsize(latest)
            except IndexError:
                latest = None
                latest_size = None
                
            # Report
            self.log.debug("=== Storage Report ===")
            self.log.debug(" directory: %s", self.directory)
            self.log.debug(" disk size: %i B", disk_total)
            self.log.debug(" disk free: %i B", disk_free)
            self.log.debug(" file count: %i", len(self._files))
            self.log.debug(" total size: %i B", total_size)
            self.log.debug(" most recent file: %s", latest)
            if latest is not None:
                self.log.debug(" most recent file size: %i B", latest_size)
            self.log.debug("===   ===")
            
            # Sleep
            if once:
                break
            time.sleep(10)


class StatusLogger(object):
    def __init__(self, log, queue, shutdown_event=None):
        self.log = log
        self.queue = queue
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        
    def _update(self):
        pass
        
    def main(self, once=False):
        while not self.shutdown_event.is_set():
            # Active operation
            is_active = False if self.queue.active is None else True
            active_filename = None
            time_left = None
            if is_active:
                active_filename = self.queue.active.filename
                time_left = self.queue.active.stop_time - self.queue.active.utcnow()
                
            # TODO: Overall system system/health
            #  What goes into this?
            #   * RX rate/missing packets?
            #   * Block processing time?
            #   * Free disk space?
            #   * Thread check?
            #   * ???
            
            # Report
            self.log.debug("=== Status Report ===")
            self.log.debug(" queue size: %i", len(self.queue))
            self.log.debug(" active operation: %s", is_active)
            if is_active:
                self.log.debug(" active filename: %s", os.path.basename(active_filename))
                self.log.debug(" active time remaining: %s", time_left)
                
            # Sleep
            if once:
                break
            time.sleep(10)

    
class GlobalLogger(object):
    def __init__(self, log, args, queue, shutdown_event=None):
        self.log = log
        self.args = args
        self.queue = queue
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self._shutdown_event = shutdown_event
        
        self.perf = PerformanceLogger(log, queue, shutdown_event=shutdown_event)
        self.storage = StorageLogger(log, args.record_directory, shutdown_event=shutdown_event)
        self.status = StatusLogger(log, queue, shutdown_event=shutdown_event)
        
    @property
    def shutdown_event(self):
        return self._shutdown_event
        
    @shutdown_event.setter
    def shutdown_event(self, event):
        self._shutdown_event = event
        for attr in ('perf', 'storage', 'status'):
            logger = getattr(self, attr, None)
            if logger is None:
                continue
            logger.shutdown_event = event
            
    def main(self):
        t_status = 0.0
        t_perf = 0.0
        t_storage = 0.0
        while not self.shutdown_event.is_set():
            # Poll
            t_now = time.time()
            if t_now - t_perf > 10.0:
                self.perf.main(once=True)
                t_perf = t_now
            if t_now - t_storage > 60.0:
                self.storage.main(once=True)
                t_storage = t_now
            if t_now - t_status > 10.0:
                self.status.main(once=True)
                t_status = t_now
                
            # Sleep
            time.sleep(10)
