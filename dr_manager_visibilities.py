#!/usr/bin/env python

from __future__ import division, print_function
try:
    range = xrange
except NameError:
    pass
    
import os
import sys
import time
import signal
import logging
import argparse
import threading

from mnc.common import *
from mnc.mcs import MonitorPoint, CommandCallbackBase, Client


def main(argv):
    parser = argparse.ArgumentParser(
                 description="Data recorder manager for slow/fast visibility data"
                 )
    parser.add_argument('-b', '--band-id', type=str, default='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16',
                        help='comma separated list of dr_visibility.py band ID number to manage')
    parser.add_argument('-l', '--logfile', type=str,
                        help='file to write logging to')
    parser.add_argument('-q', '--quick', action='store_true',
                        help='run in fast visibiltiy mode')
    parser.add_argument('-f', '--fork', action='store_true',
                        help='fork and run in the background')
    args = parser.parse_args()
    
    # Fork, if requested
    if args.fork:
        stderr = '/tmp/%s_%i.stderr' % (os.path.splitext(os.path.basename(__file__))[0], args.port)
        daemonize(stdin='/dev/null', stdout='/dev/null', stderr=stderr)
        
    # Setup logging
    log = logging.getLogger(__name__)
    logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logFormat.converter = time.gmtime
    if args.logfile is None:
        logHandler = logging.StreamHandler(sys.stdout)
    else:
        logHandler = LogFileHandler(args.logfile)
    logHandler.setFormatter(logFormat)
    log.addHandler(logHandler)
    log.setLevel(logging.DEBUG)
    
    log.info("Starting %s with PID %i", os.path.basename(__file__), os.getpid())
    log.info("Cmdline args:")
    for arg in vars(args):
        log.info("  %s: %s", arg, getattr(args, arg))
        
    # Setup the IDs to use/manage
    mcs_id = 'drv'
    if args.quick:
        mcs_id += 'f'
    else:
        mcs_id += 's'
    MANAGE_ID = []
    for band in args.band_id.split(','):
        sub_id = mcs_id+str(band)
        MANAGE_ID.append(sub_id)
        
    # Setup signal handling
    shutdown_event = setup_signal_handling([])
    
    c = Client(mcs_id)
    
    for cmd in ('ping', 'sync', 'start', 'stop'):
        cb = CommandCallbackBase(c.client)
        def wrapper(value):
            status = True
            response = {}
            for id in MANAGE_ID:
                s, r = c.send_command(id, cmd, **value)
                status &= s
                response[id] = r
            return status, response
        cb.action = wrapper
        c.set_command_callback(cmd, cb)
        
    tlast = 0.0
    while not shutdown_event.is_set():
        if time.time() - tlast > 15:
            status = "normal"
            info = ""
            first = True
            for id in MANAGE_ID:
                t0 = time.time()
                svalue = c.read_monitor_point('summary', id=id)
                ivalue = c.read_monitor_point('info', id=id)
                if svalue is None:
                    svalue = MonitorPoint("timeout", timestamp=0)
                if ivalue is None:
                    ivalue = MonitorPoint("timeout", timestamp=0)
                log.info("%s -> %s (%s) at %.0f", id, svalue.value, ivalue.value, svalue.timestamp)
                
                age = svalue.timestamp - t0
                if age > 120:
                    status = "timeout"
                if status == "error":
                    status = svalue.value
                elif svalue.value == "warning":
                    if status == "normal":
                        status = svalue.value
                        
                if not first:
                    info +="; "
                info += "%s: %s (%s)" % (id, svalue.value, ivalue.value)
                first = False
                
            ts = time.time()
            c.write_monitor_point('summary', status, timestamp=ts)
            c.write_monitor_point('info', info, timestamp=ts)
            
            tlast = ts
            
        time.sleep(2)
        
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
    
