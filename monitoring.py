import os
import sys
import glob
import time
import numpy
import shutil
import threading
from collections import deque

from bifrost.proclog import load_by_pid

from mnc.mcs import MonitorPoint, Client

__all__ = ['PerformanceLogger', 'StorageLogger', 'StatusLogger', 'GlobalLogger']


def getsize(filename):
    """
    Version of os.path.getsize that walks directories to get their total sizes.
    """
    
    if os.path.isdir(filename):
        filesize = 0
        with os.scandir(filename) as items:
            for name in items:
                if name.is_file():
                    filesize += name.stat().st_size
                elif name.is_dir():
                    filesize += getsize(name.path)
    else:
        filesize = os.path.getsize(filename)
    return filesize


class PerformanceLogger(object):
    """
    Monitoring class for logging how a Bifrost pipeline is performing.  This
    captures the maximum acquire/process/reserve times for the pipeline as well
    as the RX rate and missing packet fraction.
    """
    
    def __init__(self, log, id, queue=None, shutdown_event=None, update_interval=10):
        self.log = log
        self.id = id
        self.queue = queue
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        self.update_interval = update_interval
        
        self.client = Client(id)
        
        self._pid = os.getpid()
        self._state = deque([], 2)
        self._update()
        
    def _update(self):
        new_state = load_by_pid(self._pid)
        new_state_time = time.time()
        
        self._state.append((new_state_time,new_state))
        
    def main(self, once=False):
        """
        Main logging loop.  May be run only once with the "once" keyword set to
        True.
        """
        
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
                    acquire = max([acquire, perf['acquire_time']])
                    process = max([process, perf['process_time']])
                    reserve = max([reserve, perf['reserve_time']])
                except KeyError:
                    continue
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
                        
                rx_rate = (good1 - good0) / (self._state[1][0] - self._state[0][0])
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
            time.sleep(self.update_interval)


class StorageLogger(object):
    """
    Monitoring class for logging how storage is used by a pipeline and for enforcing
    a directory quota, if needed.
    """
    
    def __init__(self, log, id, directory, quota=None, shutdown_event=None, update_interval=10):
        self.log = log
        self.id = id
        self.directory = directory
        if quota == 0:
            quota = None
        self.quota = quota
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        self.update_interval = update_interval
        
        self.client = Client(id)
        
        self._files = []
        self._file_sizes = []
        
    def _update(self):
        self._files = glob.glob(os.path.join(self.directory, '*'))
        self._files.sort(key=lambda x: os.path.getmtime(x))
        self._file_sizes = [getsize(filename) for filename in self._files]
        
    def _manage_quota(self):
        total_size = sum(self._file_sizes)
        
        removed = []
        i = 0
        while total_size > self.quota and len(self._files) > 1:
            to_remove = self._files[i]
            
            try:
                if os.path.isdir(to_remove):
                    shutil.rmtree(to_remove)
                else:
                    os.unlink(to_remove)
                    
                removed.append(to_remove)
                del self._files[i]
                del self._file_sizes[i]
                i = 0
            except Exception as e:
                self.log.warning("Quota manager could not remove '%s': %s", to_remove, str(e))
                i += 1
                
            total_size = sum(self._file_sizes)
            
        if removed:
            self.log.debug("=== Quota Report ===")
            self.log.debug("Removed %i files", len(removed))
            self.log.debug("===   ===")
            
    def main(self, once=False):
        """
        Main logging loop.  May be run only once with the "once" keyword set to
        True.
        """
        
        while not self.shutdown_event.is_set():
            # Update the state
            self._update()
            
            # Quota management, if needed
            if self.quota is not None:
                self._manage_quota()
                
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
            total_size = sum(self._file_sizes)
            self.client.write_monitor_point('storage/active_directory',
                                            self.directory, timestamp=ts)
            self.client.write_monitor_point('storage/active_directory_size',
                                            total_size, timestamp=ts, unit='B')
            self.client.write_monitor_point('storage/active_directory_count',
                                            len(self._files), timestamp=ts)
            
            # Report
            self.log.debug("=== Storage Report ===")
            self.log.debug(" directory: %s", self.directory)
            self.log.debug(" disk size: %i B", disk_total)
            self.log.debug(" disk free: %i B", disk_free)
            self.log.debug(" file count: %i", len(self._files))
            self.log.debug(" total size: %i B", total_size)
            self.log.debug("===   ===")
            
            # Sleep
            if once:
                break
            time.sleep(self.update_interval)


class StatusLogger(object):
    """
    Monitoring class for logging the overall status of a pipeline.  This aggregates
    other monitoring points of the pipeline and uses that information to compute
    an overall state of the pipeline.
    """
    
    def __init__(self, log, id, queue, nthread=None, gulp_time=None,
                 shutdown_event=None, update_interval=10):
        self.log = log
        self.id = id
        self.queue = queue
        self.nthread = nthread
        self.gulp_time = gulp_time
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        self.update_interval = update_interval
        
        self.client = Client(id)
        
    def _update(self):
        pass
        
    @staticmethod
    def _combine_status(summary, info, new_summary, new_info):
        """
        Combine together an old summary/info pair with a new summary/info while
        respecting the hiererarchy of summaries:
         * normal is overridden by warning and error
         * warning overrides normal but not error
         * error overrides normal and warning
         * if the new summary is the same as the old summary and is either
           warning or error then the infos are combined.
        """
        
        if new_summary == 'error':
            if summary != 'error':
                info = ''
            if len(info):
                info += ', '
            summary = 'error'
            info += new_info
            
        elif new_summary == 'warning':
            if summary == 'normal':
                info = ''
            if summary == 'warning':
                if len(info):
                    info += ', '
                summary = 'warning'
                info += new_info
                
        return summary, info
        
    def main(self, once=False):
        """
        Main logging loop.  May be run only once with the "once" keyword set to
        True.
        """
        
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
            
            # Get the old summary
            old_summary = client.read_monitor_point('summary')
            old_summary = old_summary.value
            
            # Get the current metrics that matter
            active = 0
            if self.nthread is not None:
                active = threading.active_count()
            missing = self.client.read_monitor_point('bifrost/rx_missing')
            if missing is None:
                missing = MonitorPoint(0.0)
            processing = self.client.read_monitor_point('bifrost/max_process')
            if processing is None:
                processing = MonitorPoint(0.0)
            total = self.client.read_monitor_point('storage/active_disk_size')
            free = self.client.read_monitor_point('storage/active_disk_free')
            dfree = 1.0*free.value / total.value
            
            ts = min([v.timestamp for v in (missing, processing, total, free)])
            summary = 'normal'
            info = 'System operating normally'
            if self.nthread is not None:
                if active != self.nthread:
                    ## Thread check
                    new_summary = 'error'
                    new_info = "Only %i of %i threads active" % (sum(active), len(active))
                    summary, info = self._combine_status(summary, info,
                                                         new_summary, new_info)
            if dfree > 0.99:
                ## Out of space check
                new_summary = 'error'
                new_info = "No recording space (%.1f%% used)" % (dfree*100.0,)
                summary, info = self._combine_status(summary, info,
                                                     new_summary, new_info)
            elif dfree > 0.95:
                ## Low space check
                new_summary = 'warning'
                new_info = "Low recording space (%.1f%% used)" % (dfree*100.0,)
                summary, info = self._combine_status(summary, info,
                                                     new_summary, new_info)
            if missing.value > 0.10:
                ## Massive packet loss check
                new_summary = 'error'
                new_info = "Packet loss during receive >10%% (%.1f%% missing)" % (missing.value*100.0,)
                summary, info = self._combine_status(summary, info,
                                                     new_summary, new_info)
            elif missing.value > 0.01:
                ## Light packet loss check
                new_summary = 'warning'
                new_info = "Packet loss during receive >1%% (%.1f%% missing)" % (missing.value*100.0,)
                summary, info = self._combine_status(summary, info,
                                                     new_summary, new_info)
                
            if self.gulp_time is not None:
                if processing.value > self.gulp_time*1.25:
                    ## Heavy load processing time check
                    new_summary = 'error'
                    new_info = "Max. processing time at >125%% of gulp (%.3f s = %.1f%% )" % (processing.value, 100.0*processing.value/self.gulp_time)
                    summary, info = self._combine_status(summary, info,
                                                         new_summary, new_info)
                elif processing.value > self.gulp_time*1.05:
                    ## Light load processig time check
                    new_summary = 'warning'
                    new_info = "Max. processing time at >105%% of gulp (%.3f s = %.1f%% )" % (processing.value, 100.0*processing.value/self.gulp_time)
                    summary, info = self._combine_status(summary, info,
                                                         new_summary, new_info)
                    
            if summary == 'normal':
                ## De-escelation message
                if old_summary == 'warning':
                    info = 'Warning condition(s) cleared'
                elif old_summary == 'error':
                    info = 'Error condition(s) cleared'
            self.client.write_monitor_point('summary', summary, timestamp=ts)
            self.client.write_monitor_point('info', info, timestamp=ts)
                
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
            time.sleep(self.update_interval)

    
class GlobalLogger(object):
    """
    Monitoring class that wraps :py:class:`PerformanceLogger`, :py:class:`StorageLogger`,
    and :py:class:`StatusLogger` and runs their main methods as a unit.
    """
    
    def __init__(self, log, id, args, queue, quota=None, nthread=None,
                 gulp_time=None, shutdown_event=None, update_interval_perf=10,
                 update_interval_storage=60, update_interval_status=20):
        self.log = log
        self.args = args
        self.queue = queue
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self._shutdown_event = shutdown_event
        self.update_interval_perf = update_interval_perf
        self.update_interval_storage = update_interval_storage
        self.update_interval_status = update_interval_status
        self.update_internal = min([self.update_interval_perf,
                                    self.update_interval_storage,
                                    self.update_interval_status])
        
        self.id = id
        self.perf = PerformanceLogger(log, id, queue, shutdown_event=shutdown_event,
                                      update_interval=self.update_interval_perf)
        self.storage = StorageLogger(log, id, args.record_directory, quota=quota,
                                     shutdown_event=shutdown_event,
                                     update_interval=self.update_interval_storage)
        self.status = StatusLogger(log, id, queue, nthread=nthread,
                                   gulp_time=gulp_time,
                                   shutdown_event=shutdown_event,
                                   update_interval=self.update_interval_status)
        
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
        """
        Main logging loop that calls the main methods of all child loggers.
        """
        
        t_status = 0.0
        t_perf = 0.0
        t_storage = 0.0
        t_stats = 0.0
        while not self.shutdown_event.is_set():
            # Poll
            t_now = time.time()
            if t_now - t_perf > self.update_interval_perf:
                self.perf.main(once=True)
                t_perf = t_now
            if t_now - t_storage > self.update_interval_storage:
                self.storage.main(once=True)
                t_storage = t_now
            if t_now - t_status > self.update_interval_status:
                self.status.main(once=True)
                t_status = t_now
                
            # Sleep
            time.sleep(self.update_internal)
