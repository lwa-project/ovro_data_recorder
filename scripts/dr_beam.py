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
from functools import reduce
from datetime import datetime, timedelta

import zmq

from mnc.common import *
from mnc.mcs import ImageMonitorPoint, MultiMonitorPoint, Client

from ovro_data_recorder.reductions import *
from ovro_data_recorder.operations import FileOperationsQueue
from ovro_data_recorder.monitoring import GlobalLogger
from ovro_data_recorder.control import PowerBeamCommandProcessor
from ovro_data_recorder.version import version as odr_version

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


QUEUE = FileOperationsQueue()


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
        time_tag = seq0 * 2*NCHAN * navg
        hdr = {'time_tag': time_tag,
               'seq0':     seq0, 
               'chan0':    chan0,
               'cfreq0':   chan0*CHAN_BW,
               'bw':       nchan*CHAN_BW,
               'navg':     navg,
               'nbeam':    nbeam,
               'nchan':    nchan,
               'npol':     4,
               'pols':     'XX,YY,CR,CI',
               'complex':  False,
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
        seq_callback.set_pbeam(self.seq_callback)
        with UDPCapture("pbeam", self.sock, self.oring, self.nserver, self.beam0, 9000, 
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
            navg = 24
            tint = navg / CHAN_BW
            tgulp = tint * self.ntime_gulp
            nbeam = 1
            chan0 = 1234
            nchan = 16*192
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


class SpectraOp(object):
    def __init__(self, log, id, iring, ntime_gulp=250, guarantee=True, core=None):
        self.log        = log
        self.iring      = iring
        self.ntime_gulp = ntime_gulp
        self.guarantee  = guarantee
        self.core       = core
        
        self.client = Client(id)
        
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
            
            self.log.info("Spectra: Start of new sequence: %s", str(ihdr))
            
            # Setup the ring metadata and gulp sizes
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
            
            nchan_pipeline = nchan // NPIPELINE
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
                
                ## Setup and load
                idata = ispan.data_view(numpy.float32).reshape(ishape)
                
                if time.time() - last_save > 60:
                    ## Timestamp
                    tt = LWATime(time_tag, format='timetag')
                    ts = tt.datetime.strftime('%y%m%d %H:%M:%S')
                    
                    ## Average over time
                    sdata = idata.mean(axis=0)
                    
                    ## Create a diagnostic plot after suming the flags across polarization
                    ax.cla()
                    ax.plot(frange/1e6, numpy.log10(sdata[0,:,0])*10, color='#1F77B4')
                    ax.plot(frange/1e6, numpy.log10(sdata[0,:,1])*10, color='#FF7F0E')
                    ylim = ax.get_ylim()
                    for b in range(1, NPIPELINE):
                        linestyle = ':'
                        if b % 4 == 0:
                            linestyle = '--'
                        ax.vlines(frange[b*nchan_pipeline]/1e6, *ylim, linestyle=linestyle, color='black', alpha=0.2)
                    ax.set_ylim(ylim)
                    ax.set_xlim((frange[0]/1e6,frange[-1]/1e6))
                    ax.set_xlabel('Frequency [MHz]')
                    ax.set_ylabel('Power [arb. dB]')
                    ax.xaxis.set_major_locator(MultipleLocator(base=10.0))
                    fig.tight_layout()
                    
                    ## Save
                    tt = LWATime(time_tag, format='timetag')
                    mp = ImageMonitorPoint.from_figure(fig)
                    self.client.write_monitor_point('diagnostics/spectra',
                                                    mp, timestamp=tt.unix)
                    del mp
                    
                    last_save = time.time()
                    
                time_tag += navg * self.ntime_gulp * int(round(FS/CHAN_BW))
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': -1, 
                                          'process_time': process_time,})
                
        self.log.info("SpectraOp - Done")


class StatisticsOp(object):
    def __init__(self, log, id, iring, ntime_gulp=250, guarantee=True, core=None):
        self.log        = log
        self.iring      = iring
        self.ntime_gulp = ntime_gulp
        self.guarantee  = guarantee
        self.core       = core
        
        self.client = Client(id)
        
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
        
        for iseq in self.iring.read(guarantee=self.guarantee):
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            self.log.info("Statistics: Start of new sequence: %s", str(ihdr))
            
            # Setup the ring metadata and gulp sizes
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
            
            data_pols = pols.split(',')
            last_save = 0.0
            
            prev_time = time.time()
            iseq_spans = iseq.read(igulp_size)
            for ispan in iseq_spans:
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                ## Setup and load
                idata = ispan.data_view(numpy.float32).reshape(ishape)
                idata = idata.reshape(-1, npol)
                
                if time.time() - last_save > 60:
                    ## Timestamp
                    tt = LWATime(time_tag, format='timetag')
                    ts = tt.unix
                    
                    ## Run the statistics over all times/channels
                    ##  * only really works for nbeam=1
                    data_min = numpy.min(idata, axis=0)
                    data_max = numpy.max(idata, axis=0)
                    data_avg = numpy.mean(idata, axis=0)
                    
                    ## Save
                    for data,name in zip((data_min,data_avg,data_max), ('min','avg','max')):
                        value = MultiMonitorPoint(data.tolist(), timestamp=ts, field=data_pols)
                        self.client.write_monitor_point('statistics/%s' % name, value)
                        del value
                        
                    last_save = time.time()
                    
                time_tag += navg * self.ntime_gulp * int(round(FS/CHAN_BW))
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': -1, 
                                          'process_time': process_time,})
                
        self.log.info("StatisticsOp - Done")


class WriterOp(object):
    def __init__(self, log, iring, beam=1, ntime_gulp=250, swmr=False, guarantee=True, core=None):
        self.log        = log
        self.iring      = iring
        self.beam       = beam
        self.ntime_gulp = ntime_gulp
        self.swmr       = swmr
        self.guarantee  = guarantee
        self.core       = core
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        self.err_proclog = ProcLog(type(self).__name__+"/error")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        self.err_proclog.update( {'nerror':0, 'last': ''})
        
    def main(self):
        global QUEUE
        
        if self.core is not None:
            cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        was_active = False
        for iseq in self.iring.read(guarantee=self.guarantee):
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            self.log.info("Writer: Start of new sequence: %s", str(ihdr))
            
            # Setup the ring metadata and gulp sizes
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
            self.iring.resize(igulp_size, 10*igulp_size)

            norm_factor = navg
            
            first_gulp = True
            write_error_asserted = False
            write_error_counter = 0
            prev_time = time.time()
            iseq_spans = iseq.read(igulp_size)
            for ispan in iseq_spans:
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                ## On our first span, update the pipeline lag for the queue
                ## so that we start recording at the right times
                if first_gulp:
                    QUEUE.update_lag(LWATime(time_tag, format='timetag').datetime)
                    self.log.info("Current pipeline lag is %s", QUEUE.lag)
                    first_gulp = False
                    
                ## Setup and load
                idata = ispan.data_view(numpy.float32).reshape(ishape)
                try:
                    ndata[...] = idata
                except NameError:
                    ndata = idata.copy()
                ndata /= norm_factor
                
                ## Determine what to do
                active_op = QUEUE.active
                if active_op is not None:
                    ### Recording active - write
                    if not active_op.is_started:
                        self.log.info("Started operation - %s", active_op)
                        active_op.start(self.beam, chan0, navg, nchan, chan_bw, npol, pols,
                                        swmr=self.swmr)
                        was_active = True
                    try:
                        active_op.write(time_tag, ndata)
                        if write_error_asserted:
                            write_error_asserted = False
                            self.log.info("Write error de-asserted - count was %i", write_error_counter)
                            self.err_proclog.update({'nerror':0, 'last': ''})
                            write_error_counter = 0
                            
                    except Exception as e:
                        if not write_error_asserted:
                            write_error_asserted = True
                            self.log.error("Write error asserted - initial error: %s", str(e))
                            self.err_proclog.update({'nerror':1, 'last': str(e).replace(':','--')})
                        write_error_counter += 1
                        
                        if write_error_counter % 500 == 0:
                            self.log.error("Write error re-asserted - count is %i - latest error: %s", write_error_counter, str(e))
                            self.err_proclog.update( {'nerror':write_error_counter, 'last': str(e).replace(':','--')})
                            
                elif was_active:
                    ### Recording just finished - clean
                    #### Clean
                    was_active = False
                    QUEUE.clean()
                    
                    #### Close
                    self.log.info("Ended operation - %s", QUEUE.previous)
                    QUEUE.previous.stop()
                    
                time_tag += navg * self.ntime_gulp * int(round(FS/CHAN_BW))
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': -1, 
                                          'process_time': process_time,})

            try:
                del ndata
            except NameError:
                pass
                
        self.log.info("WriterOp - Done")



class AvgStreamingOp(object):
    """
    Read float32 power spectra from `iring`, average over time (axis=0),
    and stream the averaged data via ZMQ every 0.5 seconds to localhost:9798.
    Assumes gulp data reshape to (ntime_gulp, nbeam, nchan, npol).
    """
    def __init__(self, log, iring, ntime_gulp=250, guarantee=True, core=None, 
                 streaming_addr='127.0.0.1', streaming_port=9798, stream_interval=0.25):
        self.log         = log
        self.iring       = iring
        self.ntime_gulp  = ntime_gulp
        self.guarantee   = guarantee
        self.core        = core
        self.streaming_addr = streaming_addr
        self.streaming_port = streaming_port
        self.stream_interval = float(stream_interval)
        self.shutdown_event = None

        # ZMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://{self.streaming_addr}:{self.streaming_port}")
        self.log.info(f"AvgStreamingOp: ZMQ socket bound to tcp://{self.streaming_addr}:{self.streaming_port}")

        # ProcLogs (for consistency with other operations)
        self.bind_proclog      = ProcLog(type(self).__name__+"/bind")
        self.in_proclog        = ProcLog(type(self).__name__+"/in")
        self.size_proclog      = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog  = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog      = ProcLog(type(self).__name__+"/perf")

        self.in_proclog.update({'nring': 1, 'ring0': self.iring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})

    def shutdown(self):
        if self.shutdown_event:
            self.shutdown_event.set()
        if hasattr(self, 'socket'):
            self.socket.close()
        if hasattr(self, 'context'):
            self.context.term()
        self.log.info("AvgStreamingOp: Shutdown complete")

    def main(self):
        # Optional CPU pinning
        if self.core is not None:
            cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 'core0': cpu_affinity.get_core()})

        last_stream_time = 0.0
        accumulated_data = []
        accumulated_count = 0

        for iseq in self.iring.read(guarantee=self.guarantee):
            if self.shutdown_event and self.shutdown_event.is_set():
                break
                
            ihdr = json.loads(iseq.header.tostring())
            self.sequence_proclog.update(ihdr)
            self.log.info("AvgStreamingOp: Start of new sequence: %s", str(ihdr))

            time_tag    = ihdr['time_tag']
            nbeam       = ihdr['nbeam']
            nchan       = ihdr['nchan']
            npol        = ihdr['npol']
            navg        = ihdr['navg']
            chan_bw     = ihdr['bw'] / nchan
            pols        = ihdr['pols']
            pols        = pols.replace('CR', 'XY_real')
            pols        = pols.replace('CI', 'XY_imag')

            # Bytes per gulp for float32 powers
            igulp_size = self.ntime_gulp * nbeam * nchan * npol * 4
            ishape     = (self.ntime_gulp, nbeam, nchan, npol)

            prev_time = time.time()
            for ispan in iseq.read(igulp_size):
                if self.shutdown_event and self.shutdown_event.is_set():
                    break
                    
                if ispan.size < igulp_size:
                    continue  # Ignore short final gulp for consistent reshape

                t0 = time.time()
                acquire_time = t0 - prev_time
                prev_time = t0

                # Load gulp as float32 and average over time axis
                idata = ispan.data_view(numpy.float32).reshape(ishape)
                sdata = idata.mean(axis=0)   # -> (nbeam, nchan, npol)
                
                # Accumulate data for streaming
                accumulated_data.append(sdata)
                accumulated_count += 1

                # Check if it's time to stream
                now = time.time()
                if (now - last_stream_time) >= self.stream_interval and accumulated_count > 0:
                    # Average all accumulated data
                    if accumulated_count > 1:
                        avg_data = numpy.mean(accumulated_data, axis=0)
                    else:
                        avg_data = accumulated_data[0]
                    
                    t_last_block = time_tag + navg * self.ntime_gulp * int(round(FS/CHAN_BW))
                    # Prepare header with time_tag
                    stream_header = {
                        'time_tag': ihdr['time_tag'],
                        'nbeam': nbeam, # number of beams 
                        'nchan': nchan, # number of channels
                        'npol': npol, # number of polarizations
                        'timestamp': now, # time of msg creation
                        'last_block_time': str(LWATime(t_last_block, format='timetag').datetime),
                        'data_shape': avg_data.shape,
                        'data_type': '<f4',
                    }
                    
                    # Send data via ZMQ
                    try:
                        # Send header and data together
                        header_msg = json.dumps(stream_header).encode()
                        data_msg = avg_data.tobytes()
                        self.socket.send_multipart([b"data", header_msg, data_msg])
                        
                        self.log.debug("AvgStreamingOp: Streamed data with shape %s, time_tag %s", 
                                    str(avg_data.shape), str(ihdr['time_tag']))
                    except Exception as e:
                        self.log.error("AvgStreamingOp: Failed to stream data: %s", str(e))
                    
                    # Reset accumulation
                    accumulated_data = []
                    accumulated_count = 0
                    last_stream_time = now


                time_tag += navg * self.ntime_gulp * int(round(FS/CHAN_BW))
                t1 = time.time()
                process_time = t1 - prev_time
                prev_time = t1
                self.perf_proclog.update({
                    'acquire_time': acquire_time,
                    'reserve_time': -1,
                    'process_time': process_time,
                })

        self.log.info("AvgStreamingOp - Done")




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
    parser.add_argument('-c', '--cores', type=str, default='0,1,2,3',
                        help='comma separated list of cores to bind to')
    parser.add_argument('-g', '--gulp-size', type=int, default=1024,
                        help='gulp size for ring buffers')
    parser.add_argument('--swmr', action='store_true',
                        help='enable single writer/multiple reader HDF5 files')
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
    parser.add_argument('-s', '--streaming-port', type=int, default=30000,
                        help='streaming port number')
    parser.add_argument('--streaming-address', type=str, default='127.0.0.1',
                        help='streaming address')


    args = parser.parse_args()
    assert(args.gulp_size == 1024)  # Only one option
    
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
    mcs_id = 'dr%i' % args.beam
    
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
    capture_ring = Ring(name="capture", core=cores[0])
    write_ring   = Ring(name="write", core=cores[0])
    
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
                           ntime_gulp=args.gulp_size, slot_ntime=1024, core=cores.pop(0)))
    else:
        ops.append(CaptureOp(log, isock, capture_ring, NPIPELINE,
                             ntime_gulp=args.gulp_size, slot_ntime=1024, core=cores.pop(0)))

    # spectraOp is making plots, similar task can be done by AvgStreamingOp
    # ops.append(SpectraOp(log, mcs_id, capture_ring,
    #                        ntime_gulp=args.gulp_size, core=cores.pop(0)))

    ops.append(StatisticsOp(log, mcs_id, capture_ring,
                            ntime_gulp=args.gulp_size, core=cores.pop(0)))
    ops.append(WriterOp(log, capture_ring,
                        beam=args.beam, ntime_gulp=args.gulp_size,
                        swmr=args.swmr, core=cores.pop(0)))
    ops.append(GlobalLogger(log, mcs_id, args, QUEUE, quota=args.record_directory_quota,
                            threads=ops, gulp_time=args.gulp_size*24*(2*NCHAN/CLOCK)))  # Ugh, hard coded
                                
    ops.append(AvgStreamingOp(log, capture_ring,
                               ntime_gulp=args.gulp_size, core=cores.pop(0),
                               streaming_port=args.streaming_port,
                               streaming_addr=args.streaming_addr))
    
    ops.append(PowerBeamCommandProcessor(log, mcs_id, args.record_directory, QUEUE))
    
    # Setup the threads
    threads = [threading.Thread(target=op.main, name=type(op).__name__) for op in ops]
    
    # Setup signal handling
    shutdown_event = setup_signal_handling(ops)
    ops[0].shutdown_event = shutdown_event
    ops[-2].shutdown_event = shutdown_event
    ops[-1].shutdown_event = shutdown_event
    # Set shutdown event for AvgStreamingOp (index -3)
    if len(ops) >= 3:
        ops[-3].shutdown_event = shutdown_event
    
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
    
    os.system(f"kill -9 {os.getpid()}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
    
