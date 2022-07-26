#!/usr/bin/env python

from __future__ import division, print_function
try:
    range = xrange
except NameError:
    pass
    
import os
import sys
import json
import time
import uuid
import etcd3
import queue
import signal
import logging
import argparse
import threading

from mnc.common import *
from mnc.mcs import MonitorPoint, CommandCallbackBase, Client


def send_command(subsystem, command, **kwargs):
    """
    Send a command to the given subsystem and wait for a response.  The 
    arguments for the command are given as keywords.  If a response is
    received within the timeout window, that response is returned as a two-
    element tuple of (True, the response as a dictionary).  If a response
    was not received within the timeout window or another error occurred,
    return a two-element tuple of (False, sequence_id).
    """
    
    client = etcd3.client(host=ETCD_HOST, port=ETCD_PORT)
    
    if command.startswith('/'):
        command = command[1:]
        
    full_name = '/cmd/%s/%s' % (subsystem, command)
    resp_name = '/resp/'+full_name[5:]
    sequence_id = uuid.uuid1().hex
    try:
        s_id = sequence_id.decode()
    except AttributeError:
        s_id = sequence_id
    payload = {'sequence_id': sequence_id,
               'timestamp': time.time(),
               'command': command,
               'kwargs': kwargs}
    payload = json.dumps(payload)
    
    try:
        response_queue = queue.Queue()
        def callback(response, response_queue=response_queue):
            response_queue.put(response)
            
        watch_id = client.add_watch_callback(resp_name, callback)
        client.put(full_name, payload)
        
        found = None
        t0 = time.time()
        while not found and (time.time() - t0) < 0.25:
            event = response_queue.get(timeout=0.25)
            event = event.events[0]
            value = json.loads(event.value)
            if value['sequence_id'] == sequence_id:
                found = value
                break
    
    except queue.Empty:
        return False, s_id
    except Exception as e:
        return False, s_id
    finally:
        client.cancel_watch(watch_id)
        
    return True, found


def main(argv):
    parser = argparse.ArgumentParser(
                 description="Data recorder manager for slow/fast visibility data"
                 )
    parser.add_argument('-b', '--band-id', type=str, default='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16',
                        help='comma separated list of dr_visibility.py band ID numbers to manage')
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
        def wrapper(cmd=cmd, manage_id=MANAGE_ID, **value):
            status = True
            response = {}
            for id in manage_id:
                s, r = send_command(id, cmd, **value)
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
    
