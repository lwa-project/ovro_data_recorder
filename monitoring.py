import os
import sys
import glob
import time
import numpy
import threading
from collections import deque

from bifrost.proclog import load_by_pid

from mcs import MonitorPoint, Client

__all__ = ['PerformanceLogger', 'StorageLogger', 'StatusLogger', 'GlobalLogger']


class PerformanceLogger(object):
    def __init__(self, log, id, queue=None, shutdown_event=None):
        self.log = log
        self.id = id
        self.queue = queue
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        
        self.client = Client(id)
        
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
            ts = time.time()
            lag = None
            if self.queue is not None:
                lag = self.queue.lag.total_seconds()
                self.client.write_monitor_point('bifrost/pipeline_lag',
                                                lag, timestamp=ts, unit='s')
                
            # Find the maximum acquire/process/reserve times
            ts = time.time()
            acquire, process, reserve = 0.0, 0.0, 0.0
            for block,contents in self._state[1][1].items():
                try:
                    perf = contents['perf']
                except KeyError:
                    continue
                acquire = max([acquire, perf['acquire_time']])
                process = max([process, perf['process_time']])
                reserve = max([reserve, perf['reserve_time']])
            self.client.write_monitor_point('bifrost/max_acquire',
                                            acquire, timestamp=ts, unit='s')
            self.client.write_monitor_point('bifrost/max_process',
                                            process, timestamp=ts, unit='s')
            self.client.write_monitor_point('bifrost/max_reserve',
                                            reserve, timestamp=ts, unit='s')
            
            # Estimate the data rate and current missing data fracation
            rx_valid, rx_rate, missing_fraction = False, 0.0, 0.0
            good0, late0, missing0 = 0, 0, 0
            good1, late1, missing1 = 0, 0, 0
            try:
                ts = time.time()
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
                        
                rx_rate = (good1 - good1) / (self._state[1][0] - self._state[0][0])
                missing_fraction = (missing1 - missing0) / (good1 - good0 + missing1 - missing0)
                
                self.client.write_monitor_point('bifrost/rx_rate',
                                                rx_rate, timestamp=ts, unit='B/s')
                self.client.write_monitor_point('bifrost/rx_missing',
                                                missing_fraction, timestamp=ts)
                
            except (KeyError, IndexError, ZeroDivisionError):
                rx_valid = False
                
            # Load average
            ts = time.time()
            try:
                one, five, fifteen = os.getloadavg()
                self.client.write_monitor_point('system/load_average/one_minute',
                                                one, timestamp=ts)
                self.client.write_monitor_point('system/load_average/five_minute',
                                                five, timestamp=ts)
                self.client.write_monitor_point('system/load_average/fifteen_minute',
                                                fifteen, timestamp=ts)
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
    def __init__(self, log, id, directory, shutdown_event=None):
        self.log = log
        self.id = id
        self.directory = directory
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        
        self.client = Client(id)
        
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
            ts = time.time()
            st = os.statvfs(self.directory)
            disk_free = st.f_bavail * st.f_frsize
            disk_total = st.f_blocks * st.f_frsize
            self.client.write_monitor_point('storage/active_disk_size',
                                            disk_total, timestamp=ts, unit='B')
            self.client.write_monitor_point('storage/active_disk_free',
                                            disk_free, timestamp=ts, unit='B')
            
            # Find the total size of all files
            ts = time.time()
            total_size = sum([os.path.getsize(f) for f in self._files])
            self.client.write_monitor_point('storage/active_directory',
                                            self.directory, timestamp=ts)
            self.client.write_monitor_point('storage/active_directory_size',
                                            total_size, timestamp=ts, unit='B')
            self.client.write_monitor_point('storage/active_directory_count',
                                            len(self._files), timestamp=ts)
            
            # Log the last 100 files
            subfiles = self._files[-100:]
            nsubfile = len(subfiles)
            for i,subfile in enumerate(subfiles):
                self.client.write_monitor_point('storage/files/name_%i' % i,
                                                subfile, timestamp=ts)
                self.client.write_monitor_point('storage/files/size_%i' % i,
                                                os.path.getsize(subfile), timestamp=ts, unit='B')
            for i in range(nsubfile, 100):
                self.client.remove_monitor_point('storage/files/name_%i' % i)
                self.client.remove_monitor_point('storage/files/size_%i' % i)
                
            # Find the most recent file
            ts = time.time()
            try:
                latest = self._files[-1]
                latest_size = os.path.getsize(latest)
            except IndexError:
                latest = None
                latest_size = 0
            self.client.write_monitor_point('storage/active_file',
                                            latest, timestamp=ts)
            self.client.write_monitor_point('storage/active_file_size',
                                            latest_size, timestamp=ts, unit='B')
            
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
    def __init__(self, log, id, queue, shutdown_event=None):
        self.log = log
        self.id = id
        self.queue = queue
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        
        self.client = Client(id)
        
    def _update(self):
        pass
        
    def main(self, once=False):
        while not self.shutdown_event.is_set():
            # Active operation
            ts = time.time()
            is_active = False if self.queue.active is None else True
            active_filename = None
            time_left = None
            if is_active:
                active_filename = self.queue.active.filename
                time_left = self.queue.active.stop_time - self.queue.active.utcnow()
            self.client.write_monitor_point('op-type', active_filename, timestamp=ts)
            self.client.write_monitor_point('op-tag', active_filename, timestamp=ts)
            
            # TODO: Overall system system/health
            #  What goes into this?
            #   * RX rate/missing packets?
            #   * Block processing time?
            #   * Free disk space?
            #   * Thread check?
            #   * ???
            missing = self.client.read_monitor_point('bifrost/rx_missing')
            if missing is None:
                missing = MonitorPoint(0.0)
            processing = self.client.read_monitor_point('bifrost/max_process')
            total = self.client.read_monitor_point('storage/active_disk_size')
            free = self.client.read_monitor_point('storage/active_disk_free')
            dfree = 1.0*free.value / total.value
            ts = min([v.timestamp for v in (missing, processing, total, free)])
            if dfree < 0.99 and missing.value < 0.01:
                self.client.write_monitor_point('summary', 'normal', timestamp=ts)
                self.client.write_monitor_point('info', 'A-OK', timestamp=ts)
            elif dfree > 0.99 and missing.value < 0.01:
                self.client.write_monitor_point('summary', 'warning', timestamp=ts)
                self.client.write_monitor_point('info', "no space (%.1f%% used)" % (dfree*100.0,), timestamp=ts)
            elif dfree < 0.99 and missing.value > 0.01:
                self.client.write_monitor_point('summary', 'warning', timestamp=ts)
                self.client.write_monitor_point('info', "missing packets (%.1f%% missing)" % (missing.value*100.0,), timestamp=ts)
            else:
                self.client.write_monitor_point('summary', 'error', timestamp=ts)
                self.client.write_monitor_point('info', "it's bad", timestamp=ts)
                
            # Report
            self.log.debug("=== Status Report ===")
            self.log.debug(" queue size: %i", len(self.queue))
            self.log.debug(" active operation: %s", is_active)
            if is_active:
                self.log.debug(" active filename: %s", os.path.basename(active_filename))
                self.log.debug(" active time remaining: %s", time_left)
            self.log.debug("===   ===")
            
            # Sleep
            if once:
                break
            time.sleep(10)

    
class GlobalLogger(object):
    def __init__(self, log, id, args, queue, shutdown_event=None):
        self.log = log
        self.args = args
        self.queue = queue
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self._shutdown_event = shutdown_event
        
        self.id = id
        self.perf = PerformanceLogger(log, id, queue, shutdown_event=shutdown_event)
        self.storage = StorageLogger(log, id, args.record_directory, shutdown_event=shutdown_event)
        self.status = StatusLogger(log, id, queue, shutdown_event=shutdown_event)
        
    @property
    def shutdown_event(self):
        return self._shutdown_event
        
    @shutdown_event.setter
    def shutdown_event(self, event):
        self._shutdown_event = event
        for attr in ('perf', 'storage', 'status', 'stats'):
            logger = getattr(self, attr, None)
            if logger is None:
                continue
            logger.shutdown_event = event
            
    def main(self):
        t_status = 0.0
        t_perf = 0.0
        t_storage = 0.0
        t_stats = 0.0
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
