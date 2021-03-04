import os
import sys
import glob
import time
import threading

from bifrost.proclog import load_by_pid


class StatusWriter(object):
    def __init__(self, log, args, queue, shutdown_event=None):
        self.log = log
        self.args = args
        self.queue = queue
        
        self._pid = os.getpid()
        
        self._state = {}
        self._state_t = 0.0
        
        self._files = []
        
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        
    def main(self):
        # Initialize the Bifrost state tracking
        self._state = load_by_pid(self._pid)
        self._state_t = time.time()
        
        # Initialize the file list and total size
        self._files = self._update_file_listing()
        self._total_size = sum([f[1] for f in self._files])
        
        while not self.shutdown_event.is_set():
            # Poll
            self._state = load_by_pid(self._pid)
            self._state_t = time.time()
            
            self._files = self._update_file_listing()
            self._total_size = sum([f[1] for f in self._files])
            
            # Write a status
            a, p, r = self._get_bifrost_times()
            try:
                a = max(a.values())
                p = max(p.values())
                r = max(r.values())
            except ValueError:
                a, p, r = 0.0, 0.0, 0.0
            o = 'active' if self.queue.active else 'idle'
            with open('status.log', 'a') as fh:
                fh.write("%.0f %s %s %s %s %s\n" % (time.time(), o, self._total_size, a, p, r))
                
            # Wait
            time.sleep(5)
            
    def _update_file_listing(self):
        # Load in the current batch of filenames
        filenames = glob.glob('./*')
        filenames.sort(key=lambda x: os.path.getmtime(x))
        
        # Extract the relevant information about them:
        #  * name
        #  * size in bytes
        #  * modification time
        files = []
        for filename in filenames:
            files.append((filename, os.path.getsize(filename), os.path.getmtime(filename)))
            
        # Done
        return files
        
    def _get_bifrost_times(self):
        a, p, r = {}, {}, {}
        for block,contents in self._state.items():
            try:
                perf = contents['perf']
                a[block] = perf['acquire_time']
                p[block] = perf['process_time']
                r[block] = perf['reserve_time']
            except KeyError:
                continue
                
        return a, p, r
