import os
import glob
import time
import threading
from subprocess import Popen, DEVNULL
from collections import deque

from bifrost.proclog import load_by_pid

from mnc.mcs import MonitorPoint, Client

__all__ = ['PerformanceLogger', 'StorageLogger', 'StatusLogger', 'GlobalLogger']

MINIMUM_TO_DELETE_PATH_LENGTH = len("/data$$/slow")

def interruptable_sleep(seconds, sub_interval=0.1):
    """
    Version of sleep that breaks the `seconds` sleep period into sub-intervals
    of length `sub_interval`.
    """
    
    t0 = time.time()
    t1 = t0 + seconds
    while time.time() < t1:
        time.sleep(sub_interval)


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
        self._reset()
        self._update()
        
    def _reset(self):
        ts = time.time()
        for entry in ('pipeline_lag', 'max_acquire', 'max_process', 'max_reserve'):
            self.client.write_monitor_point(f"bifrost/{entry}",
                                            0.0, timestamp=ts, unit='s')
        self.client.write_monitor_point('bifrost/rx_rate',
                                        0.0, timestamp=ts, unit='B/s')
        self.client.write_monitor_point('bifrost/rx_missing',
                                        0.0, timestamp=ts)
        for entry in ('one_minute', 'five_minute', 'fifteen_minute'):
            self.client.write_monitor_point(f"system/load_average/{entry}",
                                            0.0, timestamp=ts)
            
    def _update(self):
        new_state = load_by_pid(self._pid)
        new_state_time = time.time()
        
        self._state.append((new_state_time,new_state))
        
    def _halt(self):
        self._reset()
        
    def main(self, once=False):
        """
        Main logging loop.  May be run only once with the "once" keyword set to
        True.
        """
        
        while not self.shutdown_event.is_set():
            # Update the state
            t0 = time.time()
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
            ts = time.time()
            rx_valid, rx_rate, missing_fraction = False, 0.0, 0.0
            good0, late0, missing0 = 0, 0, 0
            good1, late1, missing1 = 0, 0, 0
            try:
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
                
            except (KeyError, IndexError, ZeroDivisionError):
                pass
                
            self.client.write_monitor_point('bifrost/rx_rate',
                                            rx_rate, timestamp=ts, unit='B/s')
            self.client.write_monitor_point('bifrost/rx_missing',
                                            missing_fraction, timestamp=ts)
            
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
            self.log.debug(" elapsed time: %.3f s", time.time()-t0)
            self.log.debug("===   ===")
            
            # Sleep
            if once:
                break
                
            t1 = time.time()
            t_sleep = max([1.0, self.update_interval - (t1 - t0)])
            interruptable_sleep(t_sleep)
            
        if not once:
            self._halt()
            self.log.info("PerformanceLogger - Done")


class StorageLogger(object):
    """
    Monitoring class for logging how storage is used by a pipeline and for enforcing
    a directory quota, if needed.
    """
    
    def __init__(self, log, id, directory, quota=None, shutdown_event=None, update_interval=3600):
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
        
        self._reset()
        
    def _reset(self):
        self._files = deque()
        self._file_sizes = deque()
        
        ts = time.time()
        self.client.write_monitor_point('storage/active_disk_size',
                                        0, timestamp=ts, unit='B')
        self.client.write_monitor_point('storage/active_disk_free',
                                        0, timestamp=ts, unit='B')
        self.client.write_monitor_point('storage/active_directory',
                                        self.directory, timestamp=ts)
        self.client.write_monitor_point('storage/active_directory_size',
                                        0, timestamp=ts, unit='B')
        self.client.write_monitor_point('storage/active_directory_count',
                                        0, timestamp=ts)
        
    def _update(self):
        self.log.debug(f"StorageLogger: Updating storage usage in {self.directory}.")
        try:
            current_files = glob.glob(os.path.join(self.directory, '*'))
            current_files.sort()    # The files should have sensible names that
                                    # reflect their creation times
            
            new_files, new_file_sizes = deque(), deque()
            for filename in current_files:
                try:
                    i = self._files.index(filename)
                    new_files.append(filename)
                    new_file_sizes.append(self._file_sizes[i])
                except ValueError:
                    size = getsize(filename)
                    new_files.append(filename)
                    new_file_sizes.append(size)
        except Exception as e:
            self.log.warning("Quota manager could not refresh the file list: %s", str(e))
        self._files = new_files
        self._file_sizes = new_file_sizes
 
    def _halt(self):
        self._reset()
        
    def _manage_quota(self):
        t0 = time.time()
        total_size = sum(self._file_sizes)
        
        to_remove = []
        to_remove_size = 0
        while total_size - to_remove_size > self.quota and len(self._files) > 1:
            fn = self._files.popleft()
            f_size = self._file_sizes.popleft()
            if (len(fn) <= len(self.directory)) or \
                (not fn.startswith(self.directory)) or \
                    (len(fn) <= MINIMUM_TO_DELETE_PATH_LENGTH):
                msg = "StorageLogger: Quota management has unexpected path to remove: %s" % fn
                self.log.error(msg)
                raise ValueError(msg)
            else:
                to_remove.append(fn)
                to_remove_size += f_size
        self.log.debug("Quota: Number of items to remove: %i", len(to_remove))
        if to_remove:
            for chunk in [to_remove[i:i+100] for i in range(0, len(to_remove), 100)]:
                remove_process = Popen(['/bin/rm', '-rf'] + chunk, stdout=DEVNULL, stderr=DEVNULL)
                while remove_process.poll() is None:
                    self.shutdown_event.wait(5)
                    if self.shutdown_event.is_set():
                        remove_process.terminate()
                        return
                self.log.debug('Quota: Removed %i items.', len(chunk))
            self.log.debug("=== Quota Report ===")
            self.log.debug(" items removed: %i", len(to_remove))
            self.log.debug(" space freed: %i B", to_remove_size)
            self.log.debug(" elapsed time: %.3f s", time.time()-t0)
            self.log.debug("===   ===")
            
    def main(self, once=False):
        """
        Main logging loop.  May be run only once with the "once" keyword set to
        True.
        """
        
        while not self.shutdown_event.is_set():
            # Update the state
            t0 = time.time()
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
            total_size = sum(self._file_sizes)
            file_count = len(self._files)
            self.client.write_monitor_point('storage/active_directory',
                                            self.directory, timestamp=ts)
            self.client.write_monitor_point('storage/active_directory_size',
                                            total_size, timestamp=ts, unit='B')
            self.client.write_monitor_point('storage/active_directory_count',
                                            file_count, timestamp=ts)
            
            # Report
            self.log.debug("=== Storage Report ===")
            self.log.debug(" directory: %s", self.directory)
            self.log.debug(" disk size: %i B", disk_total)
            self.log.debug(" disk free: %i B", disk_free)
            self.log.debug(" file count: %i", file_count)
            self.log.debug(" total size: %i B", total_size)
            self.log.debug(" elapsed time: %.3f s", time.time()-t0)
            self.log.debug("===   ===")
            
            # Quota management, if needed
            if self.quota is not None:
                self._manage_quota()
                
            # Sleep
            if once:
                break
                
            t1 = time.time()
            t_sleep = max([1.0, self.update_interval - (t1 - t0)])
            interruptable_sleep(t_sleep)
            
        if not once:
            self._halt()
            self.log.info("StorageLogger - Done")


class StatusLogger(object):
    """
    Monitoring class for logging the overall status of a pipeline.  This aggregates
    other monitoring points of the pipeline and uses that information to compute
    an overall state of the pipeline.
    """
    
    def __init__(self, log, id, queue, thread_names=None, gulp_time=None,
                 shutdown_event=None, update_interval=10):
        self.log = log
        self.id = id
        self.queue = queue
        self.thread_names = thread_names
        self.gulp_time = gulp_time
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        self.update_interval = update_interval
        
        self.client = Client(id)
        self.last_summary = 'booting'
        self._reset()
        
    def _reset(self):
        ts = time.time()
        for entry in ('op-type', 'op-tag'):
            self.client.write_monitor_point(entry, None, timestamp=ts)
            
        summary = 'booting'
        info = 'System is starting up'
        self.client.write_monitor_point('summary', summary, timestamp=ts)
        self.client.write_monitor_point('info', info, timestamp=ts)
        self.last_summary = summary
        
    def _update(self):
        pass
        
    def _halt(self):
        # Change the summary to 'shutdown' when we leave exit this class
        ts = time.time()
        for entry in ('op-type', 'op-tag'):
            self.client.write_monitor_point(entry, None, timestamp=ts)
            
        summary = 'shutdown'
        info = 'System has been shutdown'
        self.client.write_monitor_point('summary', summary, timestamp=ts)
        self.client.write_monitor_point('info', info, timestamp=ts)
        self.last_summary = summary
        
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
            ts = t0 = time.time()
            is_active = False if self.queue.active is None else True
            active_filename = None
            time_left = None
            if is_active:
                active_filename = self.queue.active.filename
                time_left = self.queue.active.stop_time - self.queue.active.utcnow()
            self.client.write_monitor_point('op-type', 'recording' if is_active else 'idle', timestamp=ts)
            self.client.write_monitor_point('op-tag', active_filename, timestamp=ts)
            
            # Get the current metrics that matter
            missing_threads = []
            if self.thread_names is not None:
                found_threads = []
                for t in threading.enumerate():
                    if t.name in self.thread_names:
                        found_threads.append(t.name)
                missing_threads = [t for t in self.thread_names if t not in found_threads]
            nfound = 0
            missing = self.client.read_monitor_point('bifrost/rx_missing')
            if missing is not None:
                nfound += 1
            else:
                missing = MonitorPoint(0.0)
            processing = self.client.read_monitor_point('bifrost/max_process')
            if processing is not None:
                nfound += 1
            else:
                processing = MonitorPoint(0.0)
            total = self.client.read_monitor_point('storage/active_disk_size')
            if total is not None:
                nfound += 1
            else:
                total = MonitorPoint(0)
            free = self.client.read_monitor_point('storage/active_disk_free')
            if free is not None:
                nfound += 1
            else:
                free = MonitorPoint(0)
            if total.value != 0:
                dfree = 1.0*free.value / total.value
            else:
                dfree = 1.0
            dused = 1.0 - dfree
            
            ts = min([v.timestamp for v in (missing, processing)])
            summary = 'normal'
            info = 'System operating normally'
            if len(missing_threads) > 0:
                ## Thread check
                ntmissing = len(self.thread_names) - nactive
                new_summary = 'error'
                new_info = "Found %i missing threads - %s" % (ntmissing, ','.join(missing_threads))
                summary, info = self._combine_status(summary, info,
                                                     new_summary, new_info)
            if dused > 0.99:
                ## Out of space check
                new_summary = 'error'
                new_info = "No recording space (%.1f%% used)" % (dused*100.0,)
                summary, info = self._combine_status(summary, info,
                                                     new_summary, new_info)
            elif dused > 0.95:
                ## Low space check
                new_summary = 'warning'
                new_info = "Low recording space (%.1f%% used)" % (dused*100.0,)
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
            elif missing.value < 0.0:
                ## Nonsensical packet loss check
                new_summary = 'error'
                new_info = "Packet loss during receive is invalid (%.1f%% missing)" % (missing.value*100.0,)
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
                    
            if nfound == 0:
                ## No self monitoring information available
                new_summary = 'error'
                new_info = "Failed to query monitoring points to determine status"
                summary, info = self._combine_status(summary, info,
                                                     new_summary, new_info)
                
            if summary == 'normal':
                ## De-escelation message
                if self.last_summary == 'warning':
                    info = 'Warning condition(s) cleared'
                elif self.last_summary == 'error':
                    info = 'Error condition(s) cleared'
            self.client.write_monitor_point('summary', summary, timestamp=ts)
            self.client.write_monitor_point('info', info, timestamp=ts)
            self.last_summary = summary
            
            # Report
            self.log.debug("=== Status Report ===")
            self.log.debug(" summary: %s", summary)
            self.log.debug(" info: %s", info)
            self.log.debug(" queue size: %i", len(self.queue))
            self.log.debug(" active operation: %s", is_active)
            if is_active:
                self.log.debug(" active filename: %s", os.path.basename(active_filename))
                self.log.debug(" active time remaining: %s", time_left)
            self.log.debug(" elapsed time: %.3f s", time.time()-t0)
            self.log.debug("===   ===")
            
            # Sleep
            if once:
                break
                
            t1 = time.time()
            t_sleep = max([1.0, self.update_interval - (t1 - t0)])
            interruptable_sleep(t_sleep)
            
        if not once:
            # If this seems like it is its own thread, call _halt
            self._halt()
            self.log.info("StatusLogger - Done")


class GlobalLogger(object):
    """
    Monitoring class that wraps :py:class:`PerformanceLogger`, :py:class:`StorageLogger`,
    and :py:class:`StatusLogger` and runs their main methods as a unit.
    """
    
    def __init__(self, log, id, args, queue, quota=None, threads=None,
                 gulp_time=None, shutdown_event=None, update_interval_perf=10,
                 update_interval_storage=3600, update_interval_status=20):
        self.log = log
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self._shutdown_event = shutdown_event
        
        thread_names = []
        self._thread_names = []
        if threads is not None:
            # Explictly provided threads to monitor
            for t in threads:
                try:
                    thread_names.append(t.name)
                except AttributeError:
                    thread_names.append(type(t).__name__)
                
        # Threads associated with this logger...
        for new_thread in (type(self).__name__, 'PerformanceLogger', 'StorageLogger', 'StatusLogger'):
            # ... with a catch to deal with potentially other instances
            name = new_thread
            name_count = 0
            while name in thread_names:
                name_count += 1
                name = new_thread+str(name_count)
            thread_names.append(name)
            self._thread_names.append(name)
            
        # Reset thread_names if we don't really have a list of threads to monitor
        if threads is None:
            thread_names = None
            
        self.perf = PerformanceLogger(log, id, queue, shutdown_event=shutdown_event,
                                      update_interval=update_interval_perf)
        self.storage = StorageLogger(log, id, args.record_directory, quota=quota,
                                     shutdown_event=shutdown_event,
                                     update_interval=update_interval_storage)
        self.status = StatusLogger(log, id, queue, thread_names=thread_names,
                                   gulp_time=gulp_time,
                                   shutdown_event=shutdown_event,
                                   update_interval=update_interval_status)
        
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
        """
        Main logging loop that calls the main methods of all child loggers.
        """
        
        # Create the per-logger threads using the pre-determined thread names
        threads = []
        threads.append(threading.Thread(target=self.perf.main, name=self._thread_names[1]))
        threads.append(threading.Thread(target=self.storage.main, name=self._thread_names[2]))
        threads.append(threading.Thread(target=self.status.main, name=self._thread_names[3]))
        
        # Start the threads
        for thread in threads:
            self.log.info(f"GlobalLogger - Starting '{thread.name}'")
            #thread.daemon = True
            thread.start()
            
        # Wait for us to finish up
        while not self._shutdown_event.is_set():
            time.sleep(1)
            
        # Done
        for thread in threads:
            self.log.info(f"GlobalLogger - Waiting on '{thread.name}' to exit")
            thread.join()
        self.log.info("GlobalLogger - Done")
