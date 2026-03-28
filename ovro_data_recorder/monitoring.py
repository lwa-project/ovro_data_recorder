import os
import glob
import time
import threading
from subprocess import Popen, DEVNULL
from collections import deque
from datetime import datetime

from bifrost.proclog import load_by_pid

from mnc.mcs import MonitorPoint, Client

from ovro_data_recorder.version import version as odr_version

__all__ = ['PerformanceLogger', 'DiskStorageLogger', 'TimeStorageLogger',
           'StatusLogger', 'WatchdogLogger', 'GlobalLogger']

MINIMUM_TO_DELETE_PATH_LENGTH = len("/data$$/slow")


def interruptable_sleep(seconds, sub_interval=0.1, shutdown_event=None):
    """
    Version of sleep that breaks the `seconds` sleep period into sub-intervals
    of length `sub_interval`.
    """
    
    t0 = time.time()
    t1 = t0 + seconds
    if shutdown_event is not None:
        while time.time() < t1 and not shutdown_event.is_set():
            time.sleep(sub_interval)
    else:
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
    
    def __init__(self, log, id, queue=None, ignore_capture=True, shutdown_event=None, update_interval=10):
        self.log = log
        self.id = id
        if queue is not None:
            if not isinstance(queue, list):
                queue = [queue,]
        self.queue = queue
        self.ignore_capture = ignore_capture
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
        self.client.write_monitor_point('bifrost/error_count', 0)
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
                max_lag = -1
                for q in self.queue:
                    lag = q.lag.total_seconds()
                    if lag > max_lag:
                        max_lag = lag
                self.client.write_monitor_point('bifrost/pipeline_lag',
                                                max_lag, timestamp=ts, unit='s')
                
            # Find the maximum acquire/process/reserve times
            ts = time.time()
            acquire, process, reserve = 0.0, 0.0, 0.0
            error_count = 0
            for block,contents in self._state[1][1].items():
                if self.ignore_capture and block.find('_capture') != -1:
                    continue
                    
                try:
                    perf = contents['perf']
                    acquire = max([acquire, perf['acquire_time']])
                    process = max([process, perf['process_time']])
                    reserve = max([reserve, perf['reserve_time']])
                except KeyError:
                    pass
                    
                try:
                    err = contents['error']
                    error_count += err['nerror']
                except KeyError:
                    pass
                    
            self.client.write_monitor_point('bifrost/max_acquire',
                                            acquire, timestamp=ts, unit='s')
            self.client.write_monitor_point('bifrost/max_process',
                                            process, timestamp=ts, unit='s')
            self.client.write_monitor_point('bifrost/max_reserve',
                                            reserve, timestamp=ts, unit='s')
            self.client.write_monitor_point('bifrost/error_count',
                                            error_count, timestamp=ts)
            
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
            interruptable_sleep(t_sleep, shutdown_event=self.shutdown_event)
            
        if not once:
            self._halt()
            self.log.info("PerformanceLogger - Done")


class DiskStorageLogger(object):
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
        self.log.debug(f"DiskStorageLogger: Updating storage usage in {self.directory}.")
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
                msg = "DiskStorageLogger: Quota management has unexpected path to remove: %s" % fn
                self.log.error(msg)
                raise ValueError(msg)
            else:
                to_remove.append(fn)
                to_remove_size += f_size
        self.log.debug("Quota: Number of items to remove: %i", len(to_remove))
        if to_remove:
            batch = 0
            for chunk in [to_remove[i:i+100] for i in range(0, len(to_remove), 100)]:
                batch += 1
                try:
                    remove_process = Popen(['/bin/rm', '-rf'] + chunk, stdout=DEVNULL, stderr=DEVNULL)
                    while remove_process.poll() is None:
                        self.shutdown_event.wait(20)
                        if self.shutdown_event.is_set():
                            remove_process.kill()
                            self.log.warning('Quota: Failed to remove %i items - batch #%i took too long, giving up', len(chunk), batch)
                            return
                    self.log.debug('Quota: Removed %i items.', len(chunk))
                except OSError as e:
                    self.log.warning('Quota: Failed to remove %i items - %s', len(chunk), str(e))
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
            try:
                st = os.statvfs(self.directory)
                disk_free = st.f_bavail * st.f_frsize
                disk_total = st.f_blocks * st.f_frsize
            except OSError as e:
                self.log.warning(f"Failed to statvfs '{self.directory}': {str(e)}")
                disk_free = disk_total = 0
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
            interruptable_sleep(t_sleep, shutdown_event=self.shutdown_event)
            
        if not once:
            self._halt()
            self.log.info("DiskStorageLogger - Done")


class TimeStorageLogger(object):
    """
    Monitoring class for logging how storage is used by a pipeline and for enforcing
    a time-based directory quota, if needed.
    
    ..note:: This function assumes the following directory structure:
     * directory
     * directory/YYYY-MM-DD
     * directory/YYYY-MM-DD/HH
     * directory/YYYY-MM-DD/HH/<data>
    Quota managment is done based on the naming of the YYYY-MM-DD and HH
    directories and deletions are done at the YYYY-MM-DD and HH levels.
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
        self._file_ages = deque()
        
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
        
    def _update(self, frequency_Hz=None):
        active_dir = self.directory
        if frequency_Hz is not None:
            active_dir = os.path.join(active_dir, f"{frequency_Hz/1e6:.0f}MHz")
            
        self.log.debug(f"TimeStorageLogger: Updating storage usage in {active_dir}.")
        try:
            current_files = glob.glob(os.path.join(active_dir, '*'))
            current_files.sort()    # The files should have sensible names that
                                    # reflect their creation times
            
            t_now = datetime.utcnow()
            new_files, new_file_ages = deque(), deque()
            for filename in current_files:
                # For each top level YYYY-MM-DD directory, find all of its sub-
                # directories
                batch_filenames = [filename,]
                batch_filenames.extend(glob.glob(os.path.join(filename, '*')))
                
                # For each entry, come up with a global retention flag and an
                # age
                mark_as_retain = False
                batch_ages = []
                for fn in batch_filenames:
                    name = fn.replace(active_dir, '')
                    if name.startswith(os.path.sep):
                        name = name[len(os.path.sep):]
                        
                    if name.find('_retain') != -1:
                        mark_as_retain = True
                        
                    try:
                        ## YYYY-MM-DD/HH
                        fndate = datetime.strptime(name, '%Y-%m-%d/%H')
                    except ValueError:
                        try:
                            ## YYYY-MM-DD only
                            fndate = datetime.strptime(name, '%Y-%m-%d')
                            fndate = fndate.replace(hour=23)
                        except ValueError:
                            continue
                    fnage = (t_now - fndate).total_seconds()
                    batch_ages.append(fnage)
                    
                # If the global retention flag is set than that means there is
                # at least one HH sub-directory that should be retained.  Modify
                # the name of the parent YYYY-MM-DD directory to make sure it
                # isn't purged by the quota manager.
                if mark_as_retain:
                    batch_filenames[0] += '_retain'
                    
                # Update new_* with this batch
                new_files.extend(batch_filenames)
                new_file_ages.extend(batch_ages)
        except Exception as e:
            self.log.warning("Quota manager could not refresh the file list: %s", str(e))
        self._files = new_files
        self._file_ages = new_file_ages
 
    def _halt(self):
        self._reset()
        
    def _manage_quota(self):
        t0 = time.time()
        
        to_remove = []
        to_remove_oldest = 0
        for fn,fa in zip(self._files, self._file_ages):
            if fa > self.quota:
                if (len(fn) <= len(self.directory)) or \
                    (not fn.startswith(self.directory)) or \
                        (len(fn) <= MINIMUM_TO_DELETE_PATH_LENGTH):
                    msg = "TimeStorageLogger: Quota management has unexpected path to remove: %s" % fn
                    self.log.error(msg)
                    raise ValueError(msg)
                else:
                    if fn.find('_retain') == -1:
                        to_remove.append(fn)
                        if fa > to_remove_oldest:
                            to_remove_oldest = fa
        self.log.debug("Quota: Number of items to remove: %i", len(to_remove))
        if to_remove:
            batch = 0
            for chunk in [to_remove[i:i+100] for i in range(0, len(to_remove), 100)]:
                batch += 1
                try:
                    remove_process = Popen(['/bin/rm', '-rf'] + chunk, stdout=DEVNULL, stderr=DEVNULL)
                    while remove_process.poll() is None:
                        self.shutdown_event.wait(20)
                        if self.shutdown_event.is_set():
                            remove_process.kill()
                            self.log.warning('Quota: Failed to remove %i items - batch #%i took too long, giving up', len(chunk), batch)
                            return
                    self.log.debug('Quota: Removed %i items.', len(chunk))
                except OSError as e:
                    self.log.warning('Quota: Failed to remove %i items - %s', len(chunk), str(e))
            self.log.debug("=== Quota Report ===")
            self.log.debug(" items removed: %i", len(to_remove))
            self.log.debug(" oldest item removed: %.3f hr", (to_remove_oldest/3600.))
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
            active_freq = self.client.read_monitor_point('latest_frequency')
            if active_freq is not None:
                if active_freq.value is not None:
                    self._update(frequency_Hz=active_freq.value)
                    
            # Find the disk size and free space for the disk hosting the
            # directory - this should be quota-aware
            ts = time.time()
            try:
                st = os.statvfs(self.directory)
                disk_free = st.f_bavail * st.f_frsize
                disk_total = st.f_blocks * st.f_frsize
            except OSError as e:
                self.log.warning(f"Failed to statvfs '{self.directory}': {str(e)}")
                disk_free = disk_total = 0
            self.client.write_monitor_point('storage/active_disk_size',
                                            disk_total, timestamp=ts, unit='B')
            self.client.write_monitor_point('storage/active_disk_free',
                                            disk_free, timestamp=ts, unit='B')
            
            # Find the total size of all files
            ts = time.time()
            file_count = len(self._files)
            file_oldest = max(self._file_ages, default=0.0)
            file_newest = min(self._file_ages, default=0.0)
            self.client.write_monitor_point('storage/active_directory',
                                            self.directory, timestamp=ts)
            self.client.write_monitor_point('storage/active_directory_count',
                                            file_count, timestamp=ts)
            
            # Report
            self.log.debug("=== Storage Report ===")
            self.log.debug(" directory: %s", self.directory)
            self.log.debug(" disk size: %i B", disk_total)
            self.log.debug(" disk free: %i B", disk_free)
            self.log.debug(" age range: %.3f to %.3f hr", (file_newest/3600.), (file_oldest)/3600.0)
            self.log.debug(" elapsed time: %.3f s", time.time()-t0)
            self.log.debug("===   ===")
            
            # Quota management, if needed
            if self.quota is not None:
                if active_freq is not None:
                    self._manage_quota()
                    
            # Sleep
            if once:
                break
                
            t1 = time.time()
            t_sleep = max([1.0, self.update_interval - (t1 - t0)])
            interruptable_sleep(t_sleep, shutdown_event=self.shutdown_event)
            
        if not once:
            self._halt()
            self.log.info("TimeStorageLogger - Done")


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
        if not isinstance(queue, list):
            queue = [queue,]
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
        self.client.write_monitor_point('version', odr_version, timestamp=ts)
        
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
            is_active = False
            for q in self.queue:
                if q.active is not None:
                    is_active = True
                    break
            is_waiting = False
            active_filename = None
            time_left = None
            if is_active:
                for q in self.queue:
                    if q.active is not None:
                        q_active_filename = q.active.filename
                        q_time_left = q.active.stop_time - q.active.utcnow()
                        if q_active_filename[-3:] != '.ms' and q_active_filename[-7:] != '.ms.tar':
                            ## Only care about non-vis. data here
                            if not os.path.exists(q_active_filename):
                                is_waiting = True
                        try:
                            active_filename.append(q_active_filename)
                            time_left.append(q_time_left)
                        except AttributeError:
                            active_filename = [q_active_filename,]
                            time_left = [q_time_left,]
            optype = 'idle'
            if is_waiting:
                optype = 'waiting'
            elif is_active:
                optype = 'recording'
            if active_filename is not None:
                active_filename = ', '.join([os.path.basename(afn) for afn in active_filename])
            if time_left is not None:
                time_left = ', '.join([str(tl) for tl in time_left])
            self.client.write_monitor_point('op-type', optype, timestamp=ts)
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
            err_count = self.client.read_monitor_point('bifrost/error_count')
            if err_count is not None:
                nfound += 1
            else:
                err_count = MonitorPoint(0)
            total = self.client.read_monitor_point('storage/active_disk_size')
            if total is not None:
                nfound += 1
            else:
                total = MonitorPoint(0)
            loadavg = self.client.read_monitor_point('system/load_average/one_minute')
            if loadavg is not None:
                nfound += 1
            else:
                loadavg = MonitorPoint(0)
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
                ntmissing = len(missing_threads)
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
            elif err_count.value > 0:
                ## Non-zero block error count
                new_summary = 'error'
                new_info = "Non-zero block error count (%i errors)" % (err_count.value,)
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
            elif summary == 'warning':
                ## Forced logging of warnings conditions
                self.log.warning("Status report: %s", info)
                self.log.warning("One minute load average: %.2f", loadavg.value)
            elif summary == 'error':
                ## Forced logging of error conditions
                self.log.error("Status report: %s", info)
                self.log.error("One minute load average: %.2f", loadavg.value)
            self.client.write_monitor_point('summary', summary, timestamp=ts)
            self.client.write_monitor_point('info', info, timestamp=ts)
            self.last_summary = summary
            
            # Report
            self.log.debug("=== Status Report ===")
            self.log.debug(" summary: %s", summary)
            self.log.debug(" info: %s", info)
            self.log.debug(" queue size: %s", ', '.join([str(len(q)) for q in self.queue]))
            self.log.debug(" active operation: %s", is_active)
            if is_active:
                self.log.debug(" active filename: %s", active_filename)
                self.log.debug(" active time remaining: %s", time_left)
            self.log.debug(" elapsed time: %.3f s", time.time()-t0)
            self.log.debug("===   ===")
            
            # Sleep
            if once:
                break
                
            t1 = time.time()
            t_sleep = max([1.0, self.update_interval - (t1 - t0)])
            interruptable_sleep(t_sleep, shutdown_event=self.shutdown_event)
            
        if not once:
            # If this seems like it is its own thread, call _halt
            self._halt()
            self.log.info("StatusLogger - Done")


class WatchdogLogger(object):
    def __init__(self, log, id, pid, timeout=3600, shutdown_event=None, update_interval=600):
        self.log = log
        self.id = id
        self.pid = pid
        self.timeout = timeout
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        self.update_interval = update_interval
        
    def main(self, once=False):
        while not self.shutdown_event.is_set():
            t0 = time.time()
            
            try:
                client = Client()
                
                status = client.read_monitor_point('summary', self.id)
                if status is not None:
                    age = t0 - status.timestamp
                    if age > self.timeout:
                        self.log.error("Watchdog report: FAILED - summary last updated %.1f hr ago", (age/3600))
                        self.log.info("Watchdog: Triggering a restart by killing off pid %d", self.pid)
                        #os.system(f"kill {self.pid}")
                    else:
                        self.log.info("Watchdog report: OK - summary last updated %.1f min ago", (age/60))
                else:
                    self.log.error("Watchdog report: FAILED - summary poll returned None")
                    
                del client
                
            except Exception as e:
                self.log.error("Watchdog report: FAILED - %s", str(e))
                
            # Sleep
            if once:
                break
                
            t1 = time.time()
            t_sleep = max([1.0, self.update_interval - (t1 - t0)])
            interruptable_sleep(t_sleep, shutdown_event=self.shutdown_event)
            
        if not once:
            self.log.info("WatchdogLogger - Done")


class GlobalLogger(object):
    """
    Monitoring class that wraps :py:class:`PerformanceLogger`, :py:class:`DiskStorageLogger`/
    :py:class:`TimeStorageLogger`, and :py:class:`StatusLogger` and runs their
    main methods as a unit.
    """
    
    def __init__(self, log, id, args, queue, quota=None, threads=None,
                 gulp_time=None, shutdown_event=None, update_interval_perf=10,
                 update_interval_storage=3600, update_interval_status=20,
                 quota_mode='disk'):
        self.log = log
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self._shutdown_event = shutdown_event
        
        self.pid = os.getpid()
        
        SLC = {'disk': DiskStorageLogger,
               'time': TimeStorageLogger}
        try:
            SLC_thread_name = 'StorageLogger-'+quota_mode
            SLC = SLC[quota_mode]
        except KeyError:
            raise ValueError("Unknown quota managment mode '%s'" % quota_mode)
            
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
        for new_thread in (type(self).__name__, 'PerformanceLogger', SLC_thread_name, 'StatusLogger', 'WatchdogLogger'):
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
        self.storage = SLC(log, id, args.record_directory, quota=quota,
                            shutdown_event=shutdown_event,
                            update_interval=update_interval_storage)
        self.status = StatusLogger(log, id, queue, thread_names=thread_names,
                                   gulp_time=gulp_time,
                                   shutdown_event=shutdown_event,
                                   update_interval=update_interval_status)
        self.watchdog = WatchdogLogger(log, id, self.pid, timeout=3600,
                                       shutdown_event=shutdown_event,
                                       update_interval=600)
        
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
        threads.append(threading.Thread(target=self.watchdog.main, name=self._thread_names[4]))
        
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
