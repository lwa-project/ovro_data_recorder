#!/usr/env python

from __future__ import division, print_function
try:
    range = xrange
except NameError:
    pass

import os
import sys
import h5py
import json
import time
import numpy
import ctypes
import signal
import logging
import argparse
import threading
from functools import reduce
from datetime import datetime, timedelta

from common import *
from reductions import *
from filewriter import HDF5Writer
from operations import OperationsQueue
from monitoring import GlobalLogger
from control import BeamCommandProcessor

from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.packet_capture import PacketCaptureCallback, UDPCapture, DiskReader
from bifrost.ring import Ring
import bifrost.affinity as cpu_affinity
import bifrost.ndarray as BFArray
from bifrost.ndarray import copy_array
from bifrost.libbifrost import bf
from bifrost.proclog import ProcLog
from bifrost.memory import memcpy as BFMemCopy, memset as BFMemSet
from bifrost import asarray as BFAsArray


QUEUE = OperationsQueue()


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
        #print("++++++++++++++++ seq0     =", seq0)
        #print("                 time_tag =", time_tag)
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
        seq_callback = PacketCaptureCallback()
        seq_callback.set_pbeam(self.seq_callback)
        
        with UDPCapture("pbeam", self.sock, self.oring, self.nserver, self.beam0, 9000, 
                        self.ntime_gulp, self.slot_ntime,
                        sequence_callback=seq_callback, core=self.core) as capture:
            while not self.shutdown_event.is_set():
                status = capture.recv()
                if status in (1,4,5,6):
                    break
        del capture


class ReaderOp(object):
    def __init__(self, log, filename, oring, nserver, beam0=1, ntime_gulp=250,
                 slot_ntime=25000, shutdown_event=None, core=None):
        self.log      = log
        self.filename = filename
        self.oring    = oring
        self.nserver  = nserver
        self.beam0    = beam0
        self.ntime_gulp   = ntime_gulp
        self.slot_ntime   = slot_ntime
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        self.core    = core
        
    def shutdown(self):
        self.shutdown_event.set()
        
    def seq_callback(self, seq0, time_tag, navg, chan0, nchan, nbeam, hdr_ptr, hdr_size_ptr):
        #print("++++++++++++++++ seq0     =", seq0)
        #print("                 time_tag =", time_tag)
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
        seq_callback = PacketCaptureCallback()
        seq_callback.set_pbeam(self.seq_callback)
        
        with open(self.filename, 'rb') as fh:
            with DiskReader("pbeam_184", fh, self.oring, self.nserver, self.beam0,
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


class PlotterOp(object):
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
        if self.core is not None:
            cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        # Setup the figure
        ## Import
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        from matplotlib.ticker import MultipleLocator
        
        ## Create
        fig = plt.Figure(figsize=(6,6))
        ax = fig.gca()
        
        for iseq in self.iring.read(guarantee=self.guarantee):
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            self.log.info("Statistics: Start of new sequence: %s", str(ihdr))
            
            time_tag = ihdr['time_tag']
            navg     = ihdr['navg']
            nbeam    = ihdr['nbeam']
            chan0    = ihdr['chan0']
            nchan    = ihdr['nchan']
            chan_bw  = ihdr['bw'] / nchan
            npol     = ihdr['npol']
            pols     = ihdr['pols']
            
            igulp_size = self.ntime_gulp*nbeam*nchan*npol*4        # float32
            ishape = (self.ntime_gulp,nbeam,nchan,npol)
            
            frange = (numpy.arange(nchan) + chan0) * CHAN_BW
            last_save = 0.0
            
            prev_time = time.time()
            iseq_spans = iseq.read(igulp_size)
            for ispan in iseq_spans:
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                idata = ispan.data_view(numpy.float32).reshape(ishape)
                
                if time.time() - last_save > 60:
                    ## Average and dB
                    sdata = idata.mean(axis=0)
                    sdata = numpy.log10(sdata)*10
                    
                    ## Create a diagnostic plot after suming the flags across polarization
                    ts = time_tag / int(fS)
                    ts = datetime.datetime.utcfromtimestamp(ts)
                    ts = ts.strftime('%y%m%d %H:%M:%S')
                    
                    ax.cla()
                    ax.plot(frange/1e6, sdata[0,:,0], color='#1F77B4')
                    ax.plot(frange/1e6, sdata[0,:,1], color='#FF7F0E')
                    ax.set_xlim((frange[0]/1e6,frange[-1]/1e6))
                    ax.set_xlabel('Frequency [MHz]')
                    ax.set_ylabel('Power [arb. dB]')
                    ax.xaxis.set_major_locator(MultipleLocator(base=2.0))
                    fig.tight_layout()
                    
                    ## Save the plot
                    tt = LWATime(time_tag, format='timetag')
                    mjd, dt = tt.mjd, tt.datetime
                    mjd = int(mjd)
                    h, m, s = dt.hour, dt.minute, dt.second
                    filename = '%i_%02i%02i%02i.png' % (mjd, h, m, s)
                    canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
                    canvas.print_figure(filename)
                    
                    last_save = time.time()
                    
                time_tag += navg * self.ntime_gulp * (int(FS) // int(CHAN_BW))
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': -1, 
                                          'process_time': process_time,})


class StatisticsOp(object):
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
        
        self.data_pols = []
        self.data_min = []
        self.data_max = []
        self.data_avg = []
        
    def get_snapshot(self):
        return self.data_pols, self.data_min, self.data_max, self.data_avg
        
    def main(self):
        if self.core is not None:
            cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        for iseq in self.iring.read(guarantee=self.guarantee):
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            self.log.info("Statistics: Start of new sequence: %s", str(ihdr))
            
            time_tag = ihdr['time_tag']
            navg     = ihdr['navg']
            nbeam    = ihdr['nbeam']
            chan0    = ihdr['chan0']
            nchan    = ihdr['nchan']
            chan_bw  = ihdr['bw'] / nchan
            npol     = ihdr['npol']
            pols     = ihdr['pols']
            
            self.data_pols = pols
            
            igulp_size = self.ntime_gulp*nbeam*nchan*npol*4        # float32
            ishape = (self.ntime_gulp,nbeam,nchan,npol)
            
            prev_time = time.time()
            iseq_spans = iseq.read(igulp_size)
            for ispan in iseq_spans:
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                idata = ispan.data_view(numpy.float32).reshape(ishape)
                idata = idata.reshape(-1, npol)
                
                self.data_min = numpy.min(idata, axis=0)
                self.data_max = numpy.max(idata, axis=0)
                self.data_avg = numpy.mean(idata, axis=0)
                
                time_tag += navg * self.ntime_gulp * (int(FS) // int(CHAN_BW))
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': -1, 
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
            
            first_gulp = True 
            was_active = False
            prev_time = time.time()
            iseq_spans = iseq.read(igulp_size)
            for ispan in iseq_spans:
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                if first_gulp:
                    QUEUE.update_lag(LWATime(time_tag, format='timetag').datetime)
                    self.log.info("Current pipeline lag is %s", QUEUE.lag)
                    first_gulp = False
                    
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
                    
                time_tag += navg * self.ntime_gulp * (int(FS) // int(CHAN_BW))
                
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
    parser.add_argument('--filename', type=str,
                        help='filename containing packets to read from in offline mode')
    parser.add_argument('-b', '--beam', type=int, default=1,
                        help='beam to receive data for')
    parser.add_argument('-c', '--cores', type=str, default='0,1,2,3',
                        help='comma separated list of cores to bind to')
    parser.add_argument('-g', '--gulp-size', type=int, default=1000,
                        help='gulp size for ring buffers')
    parser.add_argument('-l', '--logfile', type=str,
                        help='file to write logging to')
    parser.add_argument('-r', '--record-directory', type=str, default=os.path.abspath('.'),
                        help='directory to save recorded files to')
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
        
    # Setup the cores and GPUs to use
    cores = [int(v, 10) for v in args.cores.split(',')]
    log.info("CPUs:         %s", ' '.join([str(v) for v in cores]))
    
    # Setup the socket, if needed
    isock = None
    if not args.offline:
        iaddr = Address(args.address, args.port)
        isock = UDPSocket()
        isock.bind(iaddr)
        
    # Setup the rings
    capture_ring = Ring(name="capture")
    write_ring   = Ring(name="write")
    
    # Setup the blocks
    ops = []
    if args.offline:
        if args.filename:
            ops.append(ReaderOp(log, args.filename, capture_ring, 16,
                                ntime_gulp=args.gulp_size, slot_ntime=1000, core=cores.pop(0)))
        else:
            ops.append(DummyOp(log, isock, capture_ring, 16,
                               ntime_gulp=args.gulp_size, slot_ntime=1000, core=cores.pop(0)))
    else:
        ops.append(CaptureOp(log, isock, capture_ring, 16,
                             ntime_gulp=args.gulp_size, slot_ntime=1000, core=cores.pop(0)))
    ops.append(PlotterOp(log, capture_ring,
                            ntime_gulp=args.gulp_size, core=cores.pop(0)))
    ops.append(StatisticsOp(log, capture_ring,
                            ntime_gulp=args.gulp_size, core=cores.pop(0)))
    ops.append(WriterOp(log, capture_ring,
                        ntime_gulp=args.gulp_size, core=cores.pop(0)))
    ops.append(GlobalLogger(log, args, QUEUE, block=ops[2]))
    ops.append(BeamCommandProcessor(log, args.record_directory, QUEUE))
    
    t_now = LWATime(datetime.utcnow() + timedelta(seconds=15), format='datetime', scale='utc')
    mjd_now = int(t_now.mjd)
    mpm_now = int((t_now.mjd - mjd_now)*86400.0*1000.0)
    ops[-1].record(json.dumps({'id': 234343423,
                               'start_mjd': mjd_now,
                               'start_mpm': mpm_now,
                               'duration_ms': 30*1000}))
    
    try:
        os.unlink(QUEUE[0].filename)
    except OSError:
        pass
        
    # Setup the threads
    threads = [threading.Thread(target=op.main) for op in ops]
    
    # Setup signal handling
    shutdown_event = setup_signal_handling(ops)
    ops[0].shutdown_event = shutdown_event
    ops[-2].shutdown_event = shutdown_event
    ops[-1].shutdown_event = shutdown_event
    
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
    
