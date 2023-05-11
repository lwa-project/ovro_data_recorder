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
    
    .. note:: This function differs from that in `mnc.mcs.Client.send_command`
              in that it makes a new etcd3 client for each call and that it
              uses etdc3 watch callbacks for the timeout handling.  This makes
              it suitable for being called from within a child thread.
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
        client.close()
        return False, s_id
    except Exception as e:
        client.close()
        return False, s_id
    finally:
        client.cancel_watch(watch_id)
        client.close()
        
    return True, found


def status_any(status, listing):
    """
    Given a status word and a list of summary values, see if any of the summaries
    match that status.
    """
    
    return any([i == status for i in listing])


def status_all(status, listing):
    """
    Given a status word and a list of summary values, see if any of the summaries
    match that status.
    """
    
    return all([i == status for i in listing])


def main(argv):
    parser = argparse.ArgumentParser(
                 description="Data recorder manager for slow/fast visibility data"
                 )
    parser.add_argument('-b', '--band-id', type=str, default='1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16',
                        help='comma separated list of dr_visibility.py band ID numbers to manage')
    parser.add_argument('-p', '--poll-interval', type=float, default=15,
                        help='band polling interval in seconds')
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
    
    # Start up
    tlast = time.time()
    summary = 'booting'
    info = 'System is starting up'
    c.write_monitor_point('summary', summary, timestamp=tlast)
    c.write_monitor_point('info', info, timestamp=tlast)
    last_summary = summary
    
    # Setup the commands
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
        
    # Enter the main polling loop
    while not shutdown_event.is_set():
        if time.time() - tlast > args.poll_interval:
            # Poll each of the sub-bands being monitored
            t0 = time.time()
            summaries, infos, actives, tags = [], [], [], []
            for id in MANAGE_ID:
                ## Poll
                svalue = c.read_monitor_point('summary', id=id)
                ivalue = c.read_monitor_point('info', id=id)
                avalue = c.read_monitor_point('op-type', id=id)
                tvalue = c.read_monitor_point('op-tag', id=id)
                ## Deal with timeouts
                if svalue is None:
                    svalue = MonitorPoint("timeout", timestamp=0)
                if ivalue is None:
                    ivalue = MonitorPoint("timeout", timestamp=0)
                if avalue is None:
                    avalue = MonitorPoint("timeout", timestamp=0)
                if tvalue is None:
                    tvalue = MonitorPoint("timeout", timestamp=0)
                ## Save
                summaries.append(svalue)
                infos.append(ivalue)
                actives.append(avalue)
                tags.append(tvalue)
                ## Report
                log.info("%s -> %s (%s) at %.0f (%.0f s ago)", id, svalue.value, ivalue.value, svalue.timestamp, time.time() - svalue.timestamp)
                log.info("%s -> operation is '%s' (%s)", id, avalue.value, tvalue.value)
            t0 = (time.time() + t0)/2.0
            
            # Get the ages
            ages = [t0 - s.timestamp for s in summaries]
            
            # Convert to simple strings
            summaries = [s.value for s in summaries]
            infos = [i.value for i in infos]
            actives = [a.value for a in actives]
            tags = [t.value for t in tags]
            
            # Create the overall op-type and op-tag values
            if status_all('recording', actives):
                op_type = 'recording'
            elif status_all('idle', actives):
                op_type = 'idle'
            else:
                nrecording = len(list(filter(lambda x: x == 'recording', actives)))
                nidle = len(list(filter(lambda x: x == 'idle', actives)))
                ntimeout = len(actives) - nrecording - nidle
                op_type = f"{nrecording} recording; {nidle} idle"
                if ntimeout > 0:
                    op_type += f"; {ntimeout} timed out"
            op_tag = '; '.join([str(t) for t in tags])
            
            tlast = time.time()
            c.write_monitor_point('op-type', op_type, timestamp=tlast)
            c.write_monitor_point('op-tag', op_tag, timestamp=tlast)
            
            # Create an overall status
            summary = 'normal'
            info = 'Systems operating normally'
            if max(ages) > 120:
                ## Any that are appear to be stale leads to an error
                stale = list(filter(lambda x: x[0] > 120, zip(ages, MANAGE_ID)))
                stale = [s[1] for s in stale]
                summary = 'error'
                info = f"{len(stale)} sub-bands have not updated in 120 s: "
                info += (','.join([str(s) for s in stale]))
                
            elif status_any('booting', summaries) \
               or status_any('shutdown', summaries) \
               or status_any('error', summaries):
                ## Any that are in booting, shutdown, or error leads to an error
                summary = 'error'
                info = ''
                for code in ('booting', 'shutdown', 'error'):
                    in_state = list(filter(lambda x: x[0] == code, zip(summaries, MANAGE_ID)))
                    in_state = [i[1] for i in in_state]
                    if len(in_state) > 0:
                        if len(info) > 0:
                            info += '; '
                        info += f"{len(in_state)} sub-bands {code}"
                        
                        info += ': '
                        for i,s,d in zip(MANAGE_ID, summaries, infos):
                            if s == code:
                                if s in ('booting', 'shutdown'):
                                    info += f"{i}; "
                                elif s == 'error':
                                    info += f"{i}={d}; "
                if info[-2:] == '; ':
                    info = info[:-2]
                    
            elif status_any('warning', summaries):
                ## Any that are in warning leads to a warning
                summary = 'warning'
                info = ''
                in_state = list(filter(lambda x: x[0] == 'warning', zip(summaries, MANAGE_ID)))
                in_state = [i[1] for i in in_state]
                if len(in_state) > 0:
                    if len(info) > 0:
                        info += '; '
                    info += f"{len(in_state)} sub-bands warning: "
                    for i,s,d in zip(MANAGE_ID, summaries, infos):
                        if s == 'warning':
                            info += f"{i}={d}; "
                if info[-2:] == '; ':
                    info = info[:-2]
                    
            if summary == 'normal':
                ## De-escelation message
                if last_summary == 'warning':
                    info = 'Warning condition(s) cleared'
                elif last_summary == 'error':
                    info = 'Error condition(s) cleared'
            tlast = time.time()
            c.write_monitor_point('summary', summary, timestamp=tlast)
            c.write_monitor_point('info', info, timestamp=tlast)
            last_summary = summary
            
        time.sleep(1)
        
    # Done
    tlast = time.time()
    summary = 'shutdown'
    info = 'System has been shutdown'
    c.write_monitor_point('summary', summary, timestamp=tlast)
    c.write_monitor_point('info', info, timestamp=tlast)
    last_summary = summary
        
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
