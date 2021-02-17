#!/usr/env python

import os
import sys
import h5py
import json
import time
import numpy
import logging
import argparse
import threading
from functools import reduce
from datetime import datetime, timedelta

from common import *
from reductions import *
from filewriter import HDF5Writer
from operations import OperationsQueue

from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.packet_capture import PacketCaptureCallback, UDPCapture
from bifrost.ring import Ring
import bifrost.affinity as cpu_affinity
import bifrost.ndarray as BFArray
from bifrost.ndarray import copy_array
from bifrost.libbifrost import bf
from bifrost.proclog import ProcLog
from bifrost.memory import memcpy as BFMemCopy, memset as BFMemSet
from bifrost import asarray as BFAsArray


QUEUE = OperationsQueue()
QUEUE.append(HDF5Writer('test.hdf5',
                        datetime.utcnow()+timedelta(seconds=15),
                        datetime.utcnow()+timedelta(seconds=30),
                        reduction=XXYYCRCI(1, 4)))


class CaptureOp(object):
    def __init__(self, log, sock, oring, nserver, beam0=1, ntime_gulp=250,
                 slot_ntime=25000, shutdown_event=None, core=None):
        self.log     = log
        self.sock    = sock
        self.oring   = oring
        self.nserver = nserver
        self.beam0   = beam0
        self.ntime_gulp   = ntime_gulp
        self.slot_ntime   = slot_ntime
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        self.core    = core
        
    def shutdown(self):
        self.shutdown_event.set()
        
    def seq_callback(self, seq0, time_tag, navg, chan0, nchan, nbeam, hdr_ptr, hdr_size_ptr):
        #print "++++++++++++++++ seq0     =", seq0
        #print "                 time_tag =", time_tag
        hdr = {'time_tag': time_tag,
               'seq0':     seq0, 
               'chan0':    chan0,
               'cfreq0':   chan0*CHAN_BW,
               'bw':       nchan*CHAN_BW,
               'navg':     navg,
               'nbeam':    nbeam,
               'npol':     4,
               'pols':     'XX,YY,CR,CI',
               'complex':  False,
               'nbit':     32}
        #print("******** HDR:", hdr)
        hdr_str = json.dumps(hdr)
        # TODO: Can't pad with NULL because returned as C-string
        #hdr_str = json.dumps(hdr).ljust(4096, '\0')
        #hdr_str = json.dumps(hdr).ljust(4096, ' ')
        header_buf = ctypes.create_string_buffer(hdr_str)
        hdr_ptr[0]      = ctypes.cast(header_buf, ctypes.c_void_p)
        hdr_size_ptr[0] = len(hdr_str)
        return 0
        
    def main(self):
        seq_callback.set_pbeam(self.seq_callback)
        with UDPCapture("pbeam", self.sock, self.oring, self.nserver, self.beam0, 9000, 
                        self.ntime_gulp, self.slot_ntime,
                        sequence_callback=seq_callback, core=self.core) as capture:
            while not self.shutdown_event.is_set():
                status = capture.recv()
                if status in (1,4,5,6):
                    break
        del capture


class DummyOp(object):
    def __init__(self, log, sock, oring, nserver, beam0=1, ntime_gulp=250,
                 slot_ntime=25000, shutdown_event=None, core=None):
        self.log     = log
        self.sock    = sock
        self.oring   = oring
        self.nserver = nserver
        self.beam0   = beam0
        self.ntime_gulp   = ntime_gulp
        self.slot_ntime   = slot_ntime
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        self.core    = core
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        
    def shutdown(self):
        self.shutdown_event.set()
          
    def main(self):
        with self.oring.begin_writing() as oring:
            navg = 24
            tint = navg / CHAN_BW
            tgulp = tint * self.ntime_gulp
            nbeam = 1
            chan0 = 1234
            nchan = 16*184
            npol = 4
            
            ohdr = {'time_tag': int(int(time.time())*FS),
                    'seq0':     0, 
                    'chan0':    chan0,
                    'cfreq0':   chan0*CHAN_BW,
                    'bw':       nchan*CHAN_BW,
                    'navg':     navg,
                    'nbeam':    nbeam,
                    'nchan':    nchan,
                    'npol':     npol,
                    'pols':     'XX,YY,CR,CI',
                    'complex':  False,
                    'nbit':     32}
            ohdr_str = json.dumps(ohdr)
            
            ogulp_size = self.ntime_gulp*nbeam*nchan*npol*4      # float32
            oshape = (self.ntime_gulp,nbeam,nchan,npol)
            self.oring.resize(ogulp_size)
            
            prev_time = time.time()
            with oring.begin_sequence(time_tag=ohdr['time_tag'], header=ohdr_str) as oseq:
                while not self.shutdown_event.is_set():
                    with oseq.reserve(ogulp_size) as ospan:
                        curr_time = time.time()
                        reserve_time = curr_time - prev_time
                        prev_time = curr_time
                        
                        odata = ospan.data_view(numpy.float32).reshape(oshape)
                        odata[...] = numpy.random.randn(*oshape)
                        
                        curr_time = time.time()
                        while curr_time - prev_time < tgulp:
                            time.sleep(0.01)
                            curr_time = time.time()
                            
                    curr_time = time.time()
                    process_time = curr_time - prev_time
                    prev_time = curr_time
                    self.perf_proclog.update({'acquire_time': -1, 
                                              'reserve_time': reserve_time, 
                                              'process_time': process_time,})


class WriterOp(object):
    def __init__(self, log, iring, ntime_gulp=250, guarantee=True, core=None):
        self.log        = log
        self.iring      = iring
        self.ntime_gulp = ntime_gulp
        self.guarantee  = guarantee
        self.core       = core
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        
    def main(self):
        global QUEUE
        
        if self.core is not None:
            cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        for iseq in self.iring.read(guarantee=self.guarantee):
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            self.log.info("Writer: Start of new sequence: %s", str(ihdr))
            
            time_tag = ihdr['time_tag']
            navg     = ihdr['navg']
            nbeam    = ihdr['nbeam']
            chan0    = ihdr['chan0']
            nchan    = ihdr['nchan']
            chan_bw  = ihdr['bw'] / nchan
            npol     = ihdr['npol']
            pols     = ihdr['pols']
            pols     = pols.replace('CR', 'XY_real')
            pols     = pols.replace('CI', 'XY_imag')
            
            igulp_size = self.ntime_gulp*nbeam*nchan*npol*4        # float32
            ishape = (self.ntime_gulp,nbeam,nchan,npol)
            
            QUEUE.update_lag(timetag_to_datetime(time_tag))
            self.log.info("Current pipeline lag is %s", QUEUE.lag)
            
            was_active = False
            prev_time = time.time()
            iseq_spans = iseq.read(igulp_size)
            for ispan in iseq_spans:
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                idata = ispan.data_view(numpy.float32).reshape(ishape)
                
                if QUEUE.active is not None:
                    # Write the data
                    if not QUEUE.active.is_started:
                        self.log.info("Started operation - %s", QUEUE.active)
                        QUEUE.active.start(1, chan0, navg, nchan, chan_bw, npol, pols)
                        was_active = True
                    QUEUE.active.write(time_tag, idata)
                elif was_active:
                    # Clean the queue
                    was_active = False
                    QUEUE.clean()
                    
                    # Close it out
                    self.log.info("Ended operation - %s", QUEUE.previous)
                    QUEUE.previous.stop()
                    
                time_tag += navg * self.ntime_gulp * (int(FS) / int(CHAN_BW))
                print(time_tag)
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': -1, 
                                          'process_time': process_time,})


def main(argv):
    parser = argparse.ArgumentParser(
                 description="Data recorder for power beams"
                 )
    parser.add_argument('-a', '--address', type=str, default='127.0.0.1',
                        help='IP address to listen to')
    parser.add_argument('-p', '--port', type=int, default=10000,
                        help='UDP port to receive data on')
    parser.add_argument('-o', '--offline', action='store_true',
                        help='run in offline using the specified file to read from')
    parser.add_argument('-b', '--beam', type=int, default=1,
                        help='beam to receive data for')
    parser.add_argument('-g', '--gulp-size', type=int, default=1024,
                        help='gulp size for ring buffers')
    parser.add_argument('-f', '--fork', action='store_true',
                        help='fork and run in the background')
    args = parser.parse_args()
    
    # Fork, if requested
    if args.fork:
        stderr = '/tmp/%s_%i.stderr' % (os.path.splitext(os.path.basename(__file__))[0], tuning)
        daemonize(stdin='/dev/null', stdout='/dev/null', stderr=stderr)
        
    # Setup logging
    log = logging.getLogger(__name__)
    logFormat = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    logFormat.converter = time.gmtime
    logHandler = logging.StreamHandler(sys.stdout)
    logHandler.setFormatter(logFormat)
    log.addHandler(logHandler)
    log.setLevel(logging.DEBUG)
     
    # Setup the socket, if needed
    isock = None
    if not args.offline:
        iaddr = Address(args.address, args.port)
        isock = UDPSocket()
        isock.bind(iaddr)
        
    # Setup the rings
    capture_ring = Ring(name="capture")
    process_ring = Ring(name="process")
    write_ring   = Ring(name="write")
    
    # Setup the blocks
    ops = []
    if args.offline:
        ops.append(DummyOp(log, isock, capture_ring, 16,
                           ntime_gulp=1000, slot_ntime=1000, core=0))
    else:
        ops.append(CaptureOp(log, isock, capture_ring, 16,
                             ntime_gulp=100, slot_ntime=1000, core=0))
    ops.append(WriterOp(log, capture_ring,
                        ntime_gulp=1000, core=2))
    try:
        os.unlink(QUEUE._queue[0].filename)
    except OSError:
        pass
        
    # Setup the threads
    threads = [threading.Thread(target=op.main) for op in ops]
    
    # Setup signal handling
    shutdown_event = setup_signal_handling(ops)
    ops[0].shutdown_event = shutdown_event
    
    # Launch!
    log.info("Launching %i thread(s)", len(threads))
    for thread in threads:
        #thread.daemon = True
        thread.start()
    while not shutdown_event.is_set():
        signal.pause()
    log.info("Shutdown, waiting for threads to join")
    for thread in threads:
        thread.join()
    log.info("All done")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
    
