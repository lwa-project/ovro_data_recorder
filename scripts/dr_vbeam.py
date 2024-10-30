#!/usr/bin/env python3

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
from collections import deque
from datetime import datetime, timedelta

from mnc.common import *
from mnc.mcs import MultiMonitorPoint, Client

from ovro_data_recorder.operations import FileOperationsQueue, DrxOperationsQueue
from ovro_data_recorder.control import RawVoltageBeamCommandProcessor
from ovro_data_recorder.version import version as odr_version

from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.packet_capture import PacketCaptureCallback, UDPVerbsCapture as UDPCapture, DiskReader
from bifrost.packet_writer import HeaderInfo, DiskWriter
from bifrost.ring import Ring
import bifrost.affinity as cpu_affinity
import bifrost.ndarray as BFArray
from bifrost.ndarray import copy_array, memset_array
from bifrost.libbifrost import bf
from bifrost.proclog import ProcLog
from bifrost import map as BFMap, asarray as BFAsArray
from bifrost.device import set_device as BFSetGPU, get_device as BFGetGPU, stream_synchronize as BFSync, set_devices_no_spin_cpu as BFNoSpinZone
BFNoSpinZone()

FILTER2BW = {1:   250000,
             2:   500000,
             3:  1000000,
             4:  2000000,
             5:  4900000,
             6:  9800000,
             7: 19600000}


FILE_QUEUE = FileOperationsQueue()
DRX_QUEUE = DrxOperationsQueue()


class CaptureOp(object):
    def __init__(self, log, sock, oring, nserver, beam0=1, ntime_gulp=250,
                 slot_ntime=25000, shutdown_event=None, core=None):
        self.log     = log
        self.sock    = sock
        self.oring   = oring
        self.nserver = nserver
        self.beam0   = beam0 - 1
        self.ntime_gulp   = ntime_gulp
        self.slot_ntime   = slot_ntime
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        self.core    = core
        
    def shutdown(self):
        self.shutdown_event.set()
        
    def seq_callback(self, seq0, chan0, nchan, nbeam, time_tag_ptr, hdr_ptr, hdr_size_ptr):
        time_tag = seq0*2*NCHAN     # Seems to be needed now
        #print("++++++++++++++++ seq0     =", seq0)
        #print("                 time_tag =", time_tag)
        hdr = {'time_tag': time_tag,
               'seq0':     seq0, 
               'chan0':    chan0,
               'cfreq0':   chan0*CHAN_BW,
               'nchan':    nchan,
               'bw':       nchan*CHAN_BW,
               'nbeam':    nbeam,
               'npol':     2,
               'complex':  True,
               'nbit':     32}
        #print("******** HDR:", hdr)
        hdr_str = json.dumps(hdr).encode()
        # TODO: Can't pad with NULL because returned as C-string
        #hdr_str = json.dumps(hdr).ljust(4096, '\0')
        #hdr_str = json.dumps(hdr).ljust(4096, ' ')
        header_buf = ctypes.create_string_buffer(hdr_str)
        hdr_ptr[0]      = ctypes.cast(header_buf, ctypes.c_void_p)
        hdr_size_ptr[0] = len(hdr_str)
        return 0
        
    def main(self):
        seq_callback = PacketCaptureCallback()
        seq_callback.set_ibeam(self.seq_callback)
        
        with UDPCapture("ibeam1", self.sock, self.oring, self.nserver, self.beam0, 9000, 
                        self.ntime_gulp, self.slot_ntime,
                        sequence_callback=seq_callback, core=self.core) as capture:
            while not self.shutdown_event.is_set():
                status = capture.recv()
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
            navg = 1
            tint = navg / CHAN_BW
            tgulp = tint * self.ntime_gulp
            nbeam = 1
            chan0 = 600
            nchan = 16*192
            npol = 2
            
            ohdr = {'time_tag': int(int(time.time())*FS),
                    'seq0':     0,
                    'chan0':    chan0,
                    'cfreq0':   chan0*CHAN_BW,
                    'bw':       nchan*CHAN_BW,
                    'nbeam':    nbeam,
                    'nchan':    nchan,
                    'npol':     npol,
                    'complex':  True,
                    'nbit':     32}
            ohdr_str = json.dumps(ohdr)
            
            ogulp_size = self.ntime_gulp*nchan*nbeam*npol*8      # complex64
            oshape = (self.ntime_gulp,nchan,nbeam,npol)
            self.oring.resize(ogulp_size)
            
            # Make some tones to look at
            t = numpy.arange(1*self.ntime_gulp*2*NCHAN) / CLOCK
            tdata = numpy.random.randn(t.size,1,2)
            for f,a0,a1 in ((20e6, 1, 2.0), (21.6e6, 2.0, 1),
                            (41e6, 4, 0.6), (44.3e6, 0.5, 3),
                            (67e6, 3, 2.2), (74.1e6, 2.5, 2)):
                p = 2*numpy.pi*numpy.random.rand()
                for b in range(nbeam):
                    tdata[:,b,0] += a0*numpy.cos(2*numpy.pi*f*t + p)
                    tdata[:,b,1] += a1*numpy.cos(2*numpy.pi*f*t + p)
            tdata = tdata.reshape(-1, 2*NCHAN, 1, 2)
            fdata = numpy.fft.fft(tdata, axis=1)[:,:NCHAN,:,:]
            fdata = fdata[:,chan0:chan0+nchan,:,:]
            fdata = fdata.astype(numpy.complex64)
            
            prev_time = time.time()
            with oring.begin_sequence(time_tag=ohdr['time_tag'], header=ohdr_str) as oseq:
                while not self.shutdown_event.is_set():
                    with oseq.reserve(ogulp_size) as ospan:
                        curr_time = time.time()
                        reserve_time = curr_time - prev_time
                        prev_time = curr_time
                        
                        odata = ospan.data_view(numpy.complex64).reshape(oshape)
                        odata[...] = fdata
                        
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


class DownSelectOp(object):
    def __init__(self, log, iring, oring, beam0=1, ntime_gulp=250, guarantee=True, core=None):
        self.log        = log
        self.iring      = iring
        self.oring      = oring
        self.beam0      = beam0
        self.ntime_gulp = ntime_gulp
        self.guarantee  = guarantee
        self.core       = core
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        
        self._pending = deque()
        self.chan0_in = 0
        self.nchan_in = 10
        self.chan0_out = 0
        self.nchan_out = int(round(FILTER2BW[1] / CHAN_BW))
        
    def updateConfig(self, hdr, time_tag, forceUpdate=False):
        global DRX_QUEUE
        
        
        # Get the current pipeline time to figure out if we need to shelve a command or not
        pipeline_time = time_tag / FS
        
        # Get the current DRX command - but only if we aren't in a forced update
        config = DRX_QUEUE.active
        if forceUpdate:
            config = None
            
        # Can we act on this configuration change now?
        if config:
            ## Pull out the beam
            beam = config[0]
            if beam != self.beam0:
                return False
            DRX_QUEUE.set_active_accepted()
            
            ## Set the configuration time - DRX commands are for the first slot in the next second
            slot = 0 / 100.0
            config_time = int(time.time()) + 1 + slot
            
            ## Is this command from the future?
            if pipeline_time < config_time:
                ### Looks like it, save it for later
                self._pending.append( (config_time, config) )
                config = None
                
                ### Is there something pending?
                try:
                    stored_time, stored_config = self._pending[0]
                    if pipeline_time >= stored_time:
                        config_time, config = self._pending.popleft()
                except IndexError:
                    pass
            else:
                ### Nope, this is something we can use now
                pass
                
        else:
            ## Is there something pending?
            try:
                stored_time, stored_config = self._pending[0]
                if pipeline_time >= stored_time:
                    config_time, config = self._pending.popleft()
            except IndexError:
                #print("No pending configuration at %.1f" % pipeline_time)
                pass
                
        if config:
            self.log.info("VBeam: New configuration received for tuning %i (delta = %.1f subslots)", config[0], (pipeline_time-config_time)*100.0)
            beam, tuning, freq, filt, gain = config
            if beam != self.beam0:
                self.log.info("VBeam: Not for this beam, skipping")
                return False
            tuning = tuning - 1
            if tuning != 0:
                self.log.info("VBeam: Not for this tuning, skipping")
                return False
                
            self.nchan_out = int(round(FILTER2BW[filt] / CHAN_BW))
            self.chan0_out = int(round(freq / CHAN_BW)) - self.nchan_out//2
            if self.chan0_out < self.chan0_in:
                self.log.warn("VBeam: Requested first channel is outside of the valid range, adjusting")
                self.chan0_out = self.chan0_in
            elif self.chan0_out + self.nchan_out > self.chan0_in + self.nchan_in:
                self.log.warn("VBeam: Requested last channel is outside of the valid range, adjusting")
                self.chan0_out = self.chan0_in + self.nchan_in - self.nchan_out
                
            return True
            
        elif forceUpdate:
            self.log.info("VBeam: New sequence configuration received")
            
            return False
        else:
            return False
            
    def main(self):
        if self.core is not None:
            cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                self.log.info("VBeam: Start of new sequence: %s", str(ihdr))
                
                self.updateConfig( ihdr, iseq.time_tag, forceUpdate=True )
                
                nbeam = ihdr['nbeam']
                chan0 = ihdr['chan0']
                nchan = ihdr['nchan']
                npol  = ihdr['npol']
                
                self.chan0_in = chan0
                self.nchan_in = nchan
                
                assert(nbeam == 1)
                assert(npol  == 2)
                
                igulp_size = self.ntime_gulp*nbeam*nchan*npol*8                # complex64
                ishape = (self.ntime_gulp,nbeam,nchan,npol)
                self.iring.resize(igulp_size, 10*igulp_size)
                
                ogulp_size = self.ntime_gulp*nbeam*self.nchan_out*npol*8       # complex64
                oshape = (self.ntime_gulp,nbeam,self.nchan_out,npol)
                self.oring.resize(ogulp_size, 10*ogulp_size)
                
                ticksPerTime = 2*NCHAN
                base_time_tag = iseq.time_tag
                
                ohdr = {}
                ohdr['nbeam']   = nbeam
                ohdr['npol']    = npol
                ohdr['complex'] = True
                ohdr['nbit']    = 32
                
                prev_time = time.time()
                iseq_spans = iseq.read(igulp_size)
                while not self.iring.writing_ended():
                    reset_sequence = False
                    
                    nchan_select = numpy.s_[self.chan0_out:self.chan0_out+self.nchan_out]
                    
                    ohdr['time_tag'] = base_time_tag
                    ohdr['chan0']    = self.chan0_out
                    ohdr['nchan']    = self.nchan_out
                    ohdr['bw']       = self.nchan_out*CHAN_BW
                    ohdr_str = json.dumps(ohdr)
                    
                    with oring.begin_sequence(time_tag=base_time_tag, header=ohdr_str) as oseq:
                        for ispan in iseq_spans:
                            if ispan.size < igulp_size:
                                continue # Ignore final gulp
                            curr_time = time.time()
                            acquire_time = curr_time - prev_time
                            prev_time = curr_time
                            
                            with oseq.reserve(ogulp_size) as ospan:
                                curr_time = time.time()
                                reserve_time = curr_time - prev_time
                                prev_time = curr_time
                                
                                ## Setup and load
                                idata = ispan.data_view(numpy.complex64).reshape(ishape)
                                odata = ospan.data_view(numpy.complex64).reshape(oshape)
                                
                                ## Prune
                                ddata = idata[:,:,nchan_select,:]
                                
                                ## Save a contiguous copy
                                odata[...] = ddata.copy()
                                
                            ## Update the base time tag
                            base_time_tag += self.ntime_gulp*ticksPerTime
                            
                            ## Check for an update to the configuration
                            if self.updateConfig( ihdr, base_time_tag, forceUpdate=False ):
                                reset_sequence = True
                                
                                ### New output size/shape
                                ngulp_size = self.ntime_gulp*nbeam*self.nchan_out*npol*8               # complex64
                                nshape = (self.ntime_gulp,nbeam,self.nchan_out,npol)
                                if ngulp_size != ogulp_size:
                                    ogulp_size = ngulp_size
                                    oshape = nshape
                                    
                                    self.oring.resize(ogulp_size, 10*ogulp_size)
                                    
                                break
                                
                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            prev_time = curr_time
                            self.perf_proclog.update({'acquire_time': acquire_time, 
                                                      'reserve_time': reserve_time, 
                                                      'process_time': process_time,})
                                                 
                    # Reset to move on to the next input sequence?
                    if not reset_sequence:
                        break


class WriterOp(object):
    def __init__(self, log, iring, beam0=1, nbeam_max=1, guarantee=True, core=None):
        self.log        = log
        self.iring      = iring
        self.beam0      = beam0
        self.nbeam_max  = nbeam_max
        self.guarantee  = guarantee
        self.core       = core
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        
    def main(self):
        global FILE_QUEUE
        
        if self.core is not None:
            cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        self.size_proclog.update({'nseq_per_gulp': 'dynamic'})
        
        desc = HeaderInfo()
        
        was_active = False
        for iseq in self.iring.read(guarantee=self.guarantee):
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            self.log.info("Writer: Start of new sequence: %s", str(ihdr))
            
            time_tag = ihdr['time_tag']
            chan0    = ihdr['chan0']
            bw       = ihdr['bw']
            npkt     = 490
            nbeam    = ihdr['nbeam']
            nchan    = ihdr['nchan']
            npol     = ihdr['npol']
            time_tag0 = iseq.time_tag // (2*NCHAN)
            time_tag  = time_tag0
            igulp_size = npkt * nbeam*nchan*npol * 8   # complex64
            
            prev_time = time.time()
            desc.set_tuning(1)
            desc.set_chan0(chan0)
            desc.set_nchan(nchan)
            desc.set_nsrc(1)
            
            first_gulp = True 
            for ispan in iseq.read(igulp_size):
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                if first_gulp:
                    first_gulp = False
                    
                shape = (npkt,nbeam,nchan*npol)
                data = ispan.data_view(numpy.complex64).reshape(shape)
                
                active_op = FILE_QUEUE.active
                if active_op is not None:
                    # Write the data
                    if not active_op.is_started:
                        self.log.info("Started operation - %s", active_op)
                        fh = active_op.start()
                        udt = DiskWriter(f"ibeam1_{nchan}", fh, core=self.core)
                        was_active = True
                        
                    time_tag_cur = time_tag
                    try:
                        udt.send(desc, time_tag_cur, 1, 1, 1, data)
                    except Exception as e:
                        print(type(self).__name__, 'Sending Error', str(e))
                        
                elif was_active:
                    # Clean the queue
                    was_active = False
                    FILE_QUEUE.clean()
                    
                    # Close it out
                    self.log.info("Ended operation - %s", FILE_QUEUE.previous)
                    del udt
                    FILE_QUEUE.previous.stop()
                    
                time_tag += npkt
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': -1, 
                                          'process_time': process_time,})


def main(argv):
    parser = argparse.ArgumentParser(
                 description="Data recorder for raw voltage beams"
                 )
    parser.add_argument('-a', '--address', type=str, default='127.0.0.1',
                        help='IP address to listen to')
    parser.add_argument('-p', '--port', type=int, default=10000,
                        help='UDP port to receive data on')
    parser.add_argument('-o', '--offline', action='store_true',
                        help='run in offline using the specified file to read from')
    parser.add_argument('-b', '--beam', type=int, default=1,
                        help='beam to receive data for')
    parser.add_argument('-c', '--cores', type=str, default='0,1,2,3,4',
                        help='comma separated list of cores to bind to')
    parser.add_argument('-g', '--gulp-size', type=int, default=1960,
                        help='gulp size for ring buffers')
    parser.add_argument('-l', '--logfile', type=str,
                        help='file to write logging to')
    parser.add_argument('--debug', action='store_true',
                        help='enable debugging messages in the log')
    parser.add_argument('-r', '--record-directory', type=str, default=os.path.abspath('.'),
                        help='directory to save recorded files to')
    parser.add_argument('-q', '--record-directory-quota', type=quota_size, default=0,
                        help='quota for the recording directory, 0 disables the quota')
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
    log.setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    log.info("Starting %s with PID %i", os.path.basename(__file__), os.getpid())
    log.info("Version: %s", odr_version)
    log.info("Cmdline args:")
    for arg in vars(args):
        log.info("  %s: %s", arg, getattr(args, arg))
        
    # Setup the subsystem ID
    mcs_id = 'drr%i' % args.beam
    
    # Setup the cores and GPUs to use
    cores = [int(v, 10) for v in args.cores.split(',')]
    log.info("CPUs:         %s", ' '.join([str(v) for v in cores]))
    
    # Setup the socket, if needed
    isock = None
    if not args.offline:
        iaddr = Address(args.address, args.port)
        isock = UDPSocket()
        isock.bind(iaddr)
        isock.timeout = 1
        
    # Setup the rings
    capture_ring = Ring(name="capture", space='cuda_host', core=cores[0])
    writer_ring = Ring(name="writer", space='cuda_host', core=cores[0])
    
    # Setup the recording directory, if needed
    if not os.path.exists(args.record_directory):
        status = os.system('mkdir -p %s' % args.record_directory)
        if status != 0:
            raise RuntimeError("Unable to create directory: %s" % args.record_directory)
    else:
        if not os.path.isdir(os.path.realpath(args.record_directory)):
            raise RuntimeError("Cannot record to a non-directory: %s" % args.record_directory)
            
    # Setup the blocks
    ops = []
    if args.offline:
        ops.append(DummyOp(log, isock, capture_ring, NPIPELINE,
                           ntime_gulp=args.gulp_size, slot_ntime=19600, core=cores.pop(0)))
    else:
        ops.append(CaptureOp(log, isock, capture_ring, NPIPELINE,
                             ntime_gulp=args.gulp_size, slot_ntime=19600, core=cores.pop(0)))
    ops.append(DownSelectOp(log, capture_ring, writer_ring, beam0=args.beam,
                            ntime_gulp=args.gulp_size, core=cores.pop(0)))
    ops.append(WriterOp(log, writer_ring, beam0=args.beam,
                        core=cores.pop(0)))
    ops.append(RawVoltageBeamCommandProcessor(log, mcs_id, args.record_directory, FILE_QUEUE, DRX_QUEUE))
    
    # Setup the threads
    threads = [threading.Thread(target=op.main, name=type(op).__name__) for op in ops]
    
    t_now = LWATime(datetime.utcnow() + timedelta(seconds=15), format='datetime', scale='utc')
    mjd_now = int(t_now.mjd)
    mpm_now = int((t_now.mjd - mjd_now)*86400.0*1000.0)
    c = Client()
    r = c.send_command(mcs_id, 'drx', beam=args.beam, tuning=1,
                       central_freq=60e6, filter=7, gain=0)
    print('III', r)
    r = c.send_command(mcs_id, 'record', beam=args.beam,
                       start_mjd=mjd_now, start_mpm=mpm_now, duration_ms=30*1000)
    print('III', r)
    
    # Setup signal handling
    shutdown_event = setup_signal_handling(ops)
    ops[0].shutdown_event = shutdown_event
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
