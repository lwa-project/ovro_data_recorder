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
from collections import deque
from datetime import datetime, timedelta

from common import *
from operations import FileOperationsQueue, DrxOperationsQueue
from monitoring import GlobalLogger
from control import VoltageBeamCommandProcessor

from bifrost.address import Address
from bifrost.udp_socket import UDPSocket
from bifrost.packet_capture import PacketCaptureCallback, UDPCapture, DiskReader
from bifrost.packet_writer import HeaderInfo, DiskWriter
from bifrost.ring import Ring
import bifrost.affinity as cpu_affinity
import bifrost.ndarray as BFArray
from bifrost.ndarray import copy_array
from bifrost.libbifrost import bf
from bifrost.proclog import ProcLog
from bifrost.fft import Fft
from bifrost.fir import Fir
from bifrost.quantize import quantize as Quantize
from bifrost.memory import memcpy as BFMemCopy, memset as BFMemSet
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
FILTER2CHAN = {1:   250000//50000, 
               2:   500000//50000, 
               3:  1000000//50000, 
               4:  2000000//50000, 
               5:  4900000//50000, 
               6:  9800000//50000, 
               7: 19600000//50000}


DRX_NSAMPLE_PER_PKT = 4096


FILE_QUEUE = FileOperationsQueue()
DRX_QUEUE = DrxOperationsQueue()


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
               'nbeam':    nbeam,
               'npol':     2,
               'complex':  True,
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
        
        with UDPCapture("ibeam1", self.sock, self.oring, self.nserver, self.beam0, 9000, 
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
            navg = 1
            tint = navg / CHAN_BW
            tgulp = tint * self.ntime_gulp
            nbeam = 1
            chan0 = 600
            nchan = 16*184
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


class ReChannelizerOp(object):
    def __init__(self, log, iring, oring, ntime_gulp=250, guarantee=True, core=None, gpu=None):
        self.log        = log
        self.iring      = iring
        self.oring      = oring
        self.ntime_gulp = ntime_gulp
        self.guarantee  = guarantee
        self.core       = core
        self.gpu        = gpu
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog  = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        
    def main(self):
        if self.core is not None:
            cpu_affinity.set_core(self.core)
        if self.gpu is not None:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
        
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                self.log.info("ReChannelizer: Start of new sequence: %s", str(ihdr))
                
                time_tag = ihdr['time_tag']
                nbeam    = ihdr['nbeam']
                chan0    = ihdr['chan0']
                nchan    = ihdr['nchan']
                chan_bw  = ihdr['bw'] / nchan
                npol     = ihdr['npol']
                
                igulp_size = self.ntime_gulp*nchan*nbeam*npol*8        # complex64
                ishape = (self.ntime_gulp,nchan,nbeam,npol)
                self.iring.resize(igulp_size, 10*igulp_size)
                
                ochan = int(round(CLOCK / 2 / 50e3))
                otime_gulp = self.ntime_gulp*NCHAN // ochan
                ogulp_size = otime_gulp*ochan*nbeam*npol*8 # complex64
                oshape = (otime_gulp,ochan,nbeam,npol)
                self.oring.resize(ogulp_size)
                
                ohdr = ihdr.copy()
                ohdr['chan0'] = 0
                ohdr['nchan'] = ochan
                ohdr['bw']    = CLOCK / 2
                ohdr_str = json.dumps(ohdr)
                
                with oring.begin_sequence(time_tag=time_tag, header=ohdr_str) as oseq:
                    prev_time = time.time()
                    iseq_spans = iseq.read(igulp_size)
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
                            
                            idata = ispan.data_view(numpy.complex64).reshape(ishape)
                            odata = ospan.data_view(numpy.complex64).reshape(oshape)
                            
                            # Pad out to the full 98 MHz bandwidth
                            try:
                                fdata[:,chan0:chan0+nchan,:,:] = idata
                            except NameError:
                                fdata = numpy.zeros((self.ntime_gulp,NCHAN,nbeam,npol), dtype=numpy.complex64)
                                fdata = BFAsArray(fdata, space='cuda_host')
                                fdata[:,chan0:chan0+nchan,:,:] = idata
                                
                            ### From here until going to the output ring we are on the GPU
                            try:
                                bdata = bdata.reshape(*fdata.shape)
                                copy_array(bdata, fdata)
                            except NameError:
                                bdata = fdata.copy(space='cuda')
                                
                            ## IFFT
                            try:
                                gdata = gdata.reshape(*bdata.shape)
                                bfft.execute(bdata, gdata, inverse=True)
                            except NameError:
                                gdata = BFArray(shape=bdata.shape, dtype=numpy.complex64, space='cuda')
                                
                                bfft = Fft()
                                bfft.init(bdata, gdata, axes=1, apply_fftshift=True)
                                bfft.execute(bdata, gdata, inverse=True)
                            gdata = gdata.reshape(otime_gulp,ochan,nbeam,npol)
                            
                            ## FFT
                            try:
                                ffft.execute(gdata, rdata, inverse=False)
                            except NameError:
                                rdata = BFArray(shape=(otime_gulp,ochan,nbeam,npol), dtype=numpy.complex64, space='cuda')
                                
                                ffft = Fft()
                                ffft.init(gdata, rdata, axes=1, apply_fftshift=True)
                                ffft.execute(gdata, rdata, inverse=False)
                                
                            ## Save
                            copy_array(odata, rdata)
                            
                        curr_time = time.time()
                        process_time = curr_time - prev_time
                        prev_time = curr_time
                        self.perf_proclog.update({'acquire_time': acquire_time, 
                                                  'reserve_time': reserve_time, 
                                                  'process_time': process_time,})
                        
            try:
                del fdata
                del gdata
                del bfft
                del rdata
                del ffft
            except NameError:
                pass


class TEngineOp(object):
    def __init__(self, log, iring, oring, beam0=1, ntime_gulp=250, guarantee=True, core=None, gpu=None):
        self.log        = log
        self.iring      = iring
        self.oring      = oring
        self.beam0      = beam0
        self.ntime_gulp = ntime_gulp
        self.guarantee  = guarantee
        self.core       = core
        self.gpu        = gpu
        
        self.bind_proclog = ProcLog(type(self).__name__+"/bind")
        self.in_proclog   = ProcLog(type(self).__name__+"/in")
        self.out_proclog   = ProcLog(type(self).__name__+"/out")
        self.size_proclog = ProcLog(type(self).__name__+"/size")
        self.sequence_proclog = ProcLog(type(self).__name__+"/sequence0")
        self.perf_proclog = ProcLog(type(self).__name__+"/perf")
        
        self.in_proclog.update(  {'nring':1, 'ring0':self.iring.name})
        self.out_proclog.update( {'nring':1, 'ring0':self.oring.name})
        self.size_proclog.update({'nseq_per_gulp': self.ntime_gulp})
        
        self._pending = deque()
        self.gain = [6, 6]
        self.rFreq = [40e6, 60e6]
        self.filt = 7
        self.nchan_out = FILTER2CHAN[self.filt]
        
        coeffs = numpy.array([ 0.0111580, -0.0074330,  0.0085684, -0.0085984,  0.0070656, -0.0035905, 
                              -0.0020837,  0.0099858, -0.0199800,  0.0316360, -0.0443470,  0.0573270, 
                              -0.0696630,  0.0804420, -0.0888320,  0.0941650,  0.9040000,  0.0941650, 
                              -0.0888320,  0.0804420, -0.0696630,  0.0573270, -0.0443470,  0.0316360, 
                              -0.0199800,  0.0099858, -0.0020837, -0.0035905,  0.0070656, -0.0085984,  
                               0.0085684, -0.0074330,  0.0111580], dtype=numpy.float64)
        
        # Setup the T-engine
        if self.gpu is not None:
            BFSetGPU(self.gpu)
        ## Metadata
        nbeam, ntune, npol = 1, 2, 2
        ## Coefficients
        coeffs.shape += (1,)
        coeffs = numpy.repeat(coeffs, nbeam*ntune*npol, axis=1)
        coeffs.shape = (coeffs.shape[0],nbeam,ntune,npol)
        self.coeffs = BFArray(coeffs, space='cuda')
        ## Phase rotator state
        phaseState = numpy.array([0,]*ntune, dtype=numpy.float64)
        self.phaseState = BFArray(phaseState, space='cuda')
        sampleCount = numpy.array([0,]*ntune, dtype=numpy.int64)
        self.sampleCount = BFArray(sampleCount, space='cuda')
        
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
            self.log.info("TEngine: New configuration received for tuning %i (delta = %.1f subslots)", config[0], (pipeline_time-config_time)*100.0)
            beam, tuning, freq, filt, gain = config
            if beam != self.beam0:
                self.log.info("TEngine: Not for this beam, skipping")
                return False
                
            self.rFreq[tuning] = freq
            self.filt = filt
            self.nchan_out = FILTER2CHAN[filt]
            self.gain[tuning] = gain
            
            chan0 = int(self.rFreq[tuning] / 50e3 + 0.5) - self.nchan_out//2
            fDiff = freq - (chan0 + 0.5*(self.nchan_out-1))*50e3 - 50e3 / 2
            self.log.info("TEngine: Tuning offset is %.3f Hz to be corrected with phase rotation", fDiff)
            
            if self.gpu is not None:
                BFSetGPU(self.gpu)
                
            phaseState = self.phaseState.copy(space='system')
            phaseState[tuning] = fDiff/(self.nchan_out*50e3)
            try:
                phaseRot = self.phaseRot.copy(space='system')
            except AttributeError:
                phaseRot = numpy.zeros((self.ntime_gulp*self.nchan_out,2), dtype=numpy.complex64)
            phaseRot[:,tuning] = numpy.exp(-2j*numpy.pi*phaseState[tuning]*numpy.arange(self.ntime_gulp*self.nchan_out, dtype=numpy.float64))
            phaseRot = phaseRot.astype(numpy.complex64)
            copy_array(self.phaseState, phaseState)
            self.phaseRot = BFAsArray(phaseRot, space='cuda')
            ACTIVE_DRX_CONFIG.set()
            
            return True
            
        elif forceUpdate:
            self.log.info("TEngine: New sequence configuration received")
            
            for tuning in (0, 1):
                try:
                    chan0 = int(self.rFreq[tuning] / 50e3 + 0.5) - self.nchan_out//2
                    fDiff = self.rFreq[tuning] - (chan0 + 0.5*(self.nchan_out-1))*50e3 - 50e3 / 2
                except AttributeError:
                    chan0 = int(40e6 / 50e3 + 0.5)
                    self.rFreq = (chan0 + 0.5*(self.nchan_out-1))*50e3 + 50e3 / 2
                    fDiff = 0.0
                self.log.info("TEngine: Tuning offset is %.3f Hz to be corrected with phase rotation", fDiff)
                
                if self.gpu is not None:
                    BFSetGPU(self.gpu)
                    
                phaseState = self.phaseState.copy(space='system')
                phaseState[tuning] = fDiff/(self.nchan_out*50e3)
                try:
                    phaseRot = self.phaseRot.copy(space='system')
                except AttributeError:
                    phaseRot = numpy.zeros((self.ntime_gulp*self.nchan_out,2), dtype=numpy.complex64)
                phaseRot[:,tuning] = numpy.exp(-2j*numpy.pi*phaseState[tuning]*numpy.arange(self.ntime_gulp*self.nchan_out, dtype=numpy.float64))
                phaseRot = phaseRot.astype(numpy.complex64)
                copy_array(self.phaseState, phaseState)
                self.phaseRot = BFAsArray(phaseRot, space='cuda')
                
            return False
            
        else:
            return False
            
    def main(self):
        if self.core is not None:
            cpu_affinity.set_core(self.core)
        if self.gpu is not None:
            BFSetGPU(self.gpu)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),
                                  'ngpu': 1,
                                  'gpu0': BFGetGPU(),})
                             
        
        with self.oring.begin_writing() as oring:
            for iseq in self.iring.read(guarantee=self.guarantee):
                ihdr = json.loads(iseq.header.tostring())
                
                self.sequence_proclog.update(ihdr)
                
                self.log.info("TEngine: Start of new sequence: %s", str(ihdr))
                
                self.rFreq[0] = 40e6
                self.rFreq[1] = 60e6
                self.updateConfig( ihdr, iseq.time_tag, forceUpdate=True )
                
                nbeam    = ihdr['nbeam']
                chan0    = ihdr['chan0']
                nchan    = ihdr['nchan']
                chan_bw  = ihdr['bw'] / nchan
                npol     = ihdr['npol']
                ntune    = 2
                
                igulp_size = self.ntime_gulp*nchan*nbeam*npol*8                # complex64
                ishape = (self.ntime_gulp,nchan,nbeam,npol)
                self.iring.resize(igulp_size, 10*igulp_size)
                
                ogulp_size = self.ntime_gulp*self.nchan_out*nbeam*ntune*npol*1       # 4+4 complex
                oshape = (self.ntime_gulp*self.nchan_out,nbeam,ntune,npol)
                self.oring.resize(ogulp_size)
                
                ticksPerTime = int(FS) // int(50e3)
                base_time_tag = iseq.time_tag
                sample_count = numpy.array([0,]*ntune, dtype=numpy.int64)
                copy_array(self.sampleCount, sample_count)
                
                tchan0 = int(self.rFreq[0] / 50e3 + 0.5) - self.nchan_out//2
                tchan1 = int(self.rFreq[1] / 50e3 + 0.5) - self.nchan_out//2
                
                ohdr = {}
                ohdr['nbeam']   = nbeam
                ohdr['ntune']   = ntune
                ohdr['npol']    = npol
                ohdr['complex'] = True
                ohdr['nbit']    = 4
                ohdr['fir_size'] = self.coeffs.shape[0]
                
                prev_time = time.time()
                iseq_spans = iseq.read(igulp_size)
                while not self.iring.writing_ended():
                    reset_sequence = False
                    
                    ohdr['time_tag'] = base_time_tag
                    ohdr['cfreq0']   = self.rFreq[0]
                    ohdr['cfreq1']   = self.rFreq[1]
                    ohdr['bw']       = self.nchan_out*50e3
                    ohdr['gain0']    = self.gain[0]
                    ohdr['gain1']    = self.gain[1]
                    ohdr['filter']   = self.filt
                    ohdr_str = json.dumps(ohdr)
                    
                    # Adjust the gain to make this ~compatible with LWA1
                    act_gain0 = self.gain[0] + 15
                    act_gain1 = self.gain[1] + 15
                    rel_gain = numpy.array([1.0, (2**act_gain0)/(2**act_gain1)], dtype=numpy.float32)
                    rel_gain = BFArray(rel_gain, space='cuda')
                    
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
                                odata = ospan.data_view(numpy.int8).reshape(oshape)
                                
                                ## Prune the data ahead of the IFFT
                                try:
                                    pdata[:,:,:,0,:] = idata[:,tchan0:tchan0+self.nchan_out,:,:]
                                    pdata[:,:,:,1,:] = idata[:,tchan1:tchan1+self.nchan_out,:,:]
                                except NameError:
                                    pshape = (self.ntime_gulp,self.nchan_out,nbeam,ntune,npol)
                                    pdata = BFArray(shape=pshape, dtype=numpy.complex64, space='cuda_host')
                                    
                                    pdata[:,:,:,0,:] = idata[:,tchan0:tchan0+self.nchan_out,:,:]
                                    pdata[:,:,:,1,:] = idata[:,tchan1:tchan1+self.nchan_out,:,:]
                                    
                                ### From here until going to the output ring we are on the GPU
                                try:
                                    copy_array(bdata, pdata)
                                except NameError:
                                    bdata = pdata.copy(space='cuda')
                                    
                                ## IFFT
                                try:
                                    gdata = gdata.reshape(*bdata.shape)
                                    bfft.execute(bdata, gdata, inverse=True)
                                except NameError:
                                    gdata = BFArray(shape=bdata.shape, dtype=numpy.complex64, space='cuda')
                                    
                                    bfft = Fft()
                                    bfft.init(bdata, gdata, axes=1, apply_fftshift=True)
                                    bfft.execute(bdata, gdata, inverse=True)
                                    
                                ## Phase rotation and output "desired gain imbalance" correction
                                gdata = gdata.reshape((-1,nbeam*ntune*npol))
                                BFMap("""
                                      auto k = (j / 2) % 2;
                                      a(i,j) *= exp(Complex<float>(r(k), -2*BF_PI_F*r(k)*fmod(g(k)*s(k), 1.0)))*b(i,k);
                                      """, 
                                      {'a':gdata, 'b':self.phaseRot, 'g':self.phaseState, 's':self.sampleCount, 'r':rel_gain},
                                      axis_names=('i','j'),
                                      shape=gdata.shape, 
                                      extra_code="#define BF_PI_F 3.141592654f")
                                gdata = gdata.reshape((-1,nbeam,ntune,npol))
                                
                                ## FIR filter
                                try:
                                    bfir.execute(gdata, fdata)
                                except NameError:
                                    fdata = BFArray(shape=gdata.shape, dtype=gdata.dtype, space='cuda')
                                    
                                    bfir = Fir()
                                    bfir.init(self.coeffs, 1)
                                    bfir.execute(gdata, fdata)
                                    
                                ## Quantization
                                try:
                                    Quantize(fdata, qdata, scale=8./(2**act_gain0 * numpy.sqrt(self.nchan_out)))
                                except NameError:
                                    qdata = BFArray(shape=fdata.shape, native=False, dtype='ci4', space='cuda')
                                    Quantize(fdata, qdata, scale=8./(2**act_gain0 * numpy.sqrt(self.nchan_out)))
                                    
                                ## Save
                                try:
                                    copy_array(tdata, qdata)
                                except NameError:
                                    tdata = qdata.copy('system')
                                odata[...] = tdata.view(numpy.int8).reshape(self.ntime_gulp*self.nchan_out,nbeam,ntune,npol)
                                
                            ## Update the base time tag
                            base_time_tag += self.ntime_gulp*ticksPerTime
                            
                            ## Update the sample counter
                            sample_count += oshape[0]
                            copy_array(self.sampleCount, sample_count)
                            
                            ## Check for an update to the configuration
                            if self.updateConfig( ihdr, base_time_tag, forceUpdate=False ):
                                reset_sequence = True
                                sample_count *= 0
                                copy_array(self.sampleCount, sample_count)
                                
                                ### New output size/shape
                                ngulp_size = ntune*self.ntime_gulp*self.nchan_out*nbeam*npol*1               # 4+4 complex
                                nshape = (ntune,self.ntime_gulp*self.nchan_out,nbeam,npol)
                                if ngulp_size != ogulp_size:
                                    ogulp_size = ngulp_size
                                    oshape = nshape
                                    
                                    self.oring.resize(ogulp_size)
                                    
                                ### Clean-up
                                try:
                                    del pdata
                                    del bdata
                                    del gdata
                                    del bfft
                                    del fdata
                                    del bfir
                                    del qdata
                                    del tdata
                                except NameError:
                                    pass
                                    
                                break
                                
                            curr_time = time.time()
                            process_time = curr_time - prev_time
                            prev_time = curr_time
                            self.perf_proclog.update({'acquire_time': acquire_time, 
                                                      'reserve_time': reserve_time, 
                                                      'process_time': process_time,})
                                                 
                    # Reset to move on to the next input sequence?
                    if not reset_sequence:
                        ## Clean-up
                        try:
                            del pdata
                            del gdata
                            del fdata
                            del qdata
                        except NameError:
                            pass
                            
                        break


class WriterOp(object):
    def __init__(self, log, iring, beam0=1, npkt_gulp=128, nbeam_max=1, ntune_max=2, guarantee=True, core=None):
        self.log        = log
        self.iring      = iring
        self.beam0      = beam0
        self.npkt_gulp  = npkt_gulp
        self.nbeam_max  = nbeam_max
        self.ntune_max  = ntune_max
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
        
        ntime_pkt     = DRX_NSAMPLE_PER_PKT
        ntime_gulp    = self.npkt_gulp * ntime_pkt
        ninput_max    = self.nbeam_max * self.ntune_max * 2
        igulp_size_max = ntime_gulp * ninput_max * 2
        self.iring.resize(igulp_size_max)
        
        self.size_proclog.update({'nseq_per_gulp': ntime_gulp})
        
        desc0 = HeaderInfo()
        desc1 = HeaderInfo()
        
        for iseq in self.iring.read(guarantee=self.guarantee):
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            self.log.info("Writer: Start of new sequence: %s", str(ihdr))
            
            time_tag = ihdr['time_tag']
            cfreq0   = ihdr['cfreq0']
            cfreq1   = ihdr['cfreq1']
            bw       = ihdr['bw']
            gain0    = ihdr['gain0']
            gain1    = ihdr['gain1']
            filt     = ihdr['filter']
            nbeam    = ihdr['nbeam']
            ntune    = ihdr['ntune']
            npol     = ihdr['npol']
            fdly     = (ihdr['fir_size'] - 1) / 2.0
            time_tag0 = iseq.time_tag
            time_tag  = time_tag0
            igulp_size = ntime_gulp*nbeam*ntune*npol
            
            # Figure out where we need to be in the buffer to be at a frame boundary
            NPACKET_SET = 4
            ticksPerSample = int(FS) // int(bw)
            toffset = int(time_tag0) // ticksPerSample
            soffset = toffset % (NPACKET_SET*int(ntime_pkt))
            if soffset != 0:
                soffset = NPACKET_SET*ntime_pkt - soffset
            boffset = soffset*nbeam*ntune*npol
            print('!!', '@', self.beam0, toffset, '->', (toffset*int(round(bw))), ' or ', soffset, ' and ', boffset, ' at ', ticksPerSample)
            
            time_tag += soffset*ticksPerSample                  # Correct for offset
            time_tag -= int(round(fdly*ticksPerSample))         # Correct for FIR filter delay
            
            prev_time = time.time()
            desc0.set_decimation(int(FS)//int(bw))
            desc1.set_decimation(int(FS)//int(bw))
            desc0.set_tuning(int(round(cfreq0 / FS * 2**32)))
            desc1.set_tuning(int(round(cfreq1 / FS * 2**32)))
            desc_src = ((1&0x7)<<3)
            
            first_gulp = True 
            was_active = False
            for ispan in iseq.read(igulp_size, begin=boffset):
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                if first_gulp:
                    FILE_QUEUE.update_lag(LWATime(time_tag, format='timetag').datetime)
                    self.log.info("Current pipeline lag is %s", FILE_QUEUE.lag)
                    first_gulp = False
                    
                shape = (-1,nbeam,ntune,npol)
                data = ispan.data_view('ci4').reshape(shape)
                
                data0 = data[:,:,0,:].reshape(-1,ntime_pkt,nbeam*npol).transpose(0,2,1).copy()
                data1 = data[:,:,1,:].reshape(-1,ntime_pkt,nbeam*npol).transpose(0,2,1).copy()
                
                if FILE_QUEUE.active is not None:
                    # Write the data
                    if not FILE_QUEUE.active.is_started:
                        self.log.info("Started operation - %s", FILE_QUEUE.active)
                        fh = FILE_QUEUE.active.start()
                        udt = DiskWriter("drx", fh, core=self.core)
                        was_active = True
                        
                    for t in range(0, data0.shape[0], NPACKET_SET):
                        time_tag_cur = time_tag + t*ticksPerSample*ntime_pkt
                        
                        try:
                            udt.send(desc0, time_tag_cur, ticksPerSample*ntime_pkt, desc_src+self.beam0, 128, 
                                     data0[t:t+NPACKET_SET,:,:])
                            udt.send(desc1, time_tag_cur, ticksPerSample*ntime_pkt, desc_src+8+self.beam0, 128, 
                                     data1[t:t+NPACKET_SET,:,:])
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
                    
                time_tag += int(ntime_gulp)*ticksPerSample
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': -1, 
                                          'process_time': process_time,})


def main(argv):
    parser = argparse.ArgumentParser(
                 description="T-Engine and data recorder for voltage beams"
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
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to bind to')
    parser.add_argument('-g', '--gulp-size', type=int, default=1960,
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
    gpus = [args.gpu for c in cores]
    log.info("CPUs:         %s", ' '.join([str(v) for v in cores]))
    log.info("GPUs:         %s", ' '.join([str(v) for v in gpus]))
    
    # Setup the socket, if needed
    isock = None
    if not args.offline:
        iaddr = Address(args.address, args.port)
        isock = UDPSocket()
        isock.bind(iaddr)
        
    # Setup the rings
    capture_ring = Ring(name="capture", space='cuda_host')
    tengine_ring = Ring(name="tengine", space='cuda_host')
    write_ring   = Ring(name="write", space='cuda_host')
    
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
    ops.append(ReChannelizerOp(log, capture_ring, tengine_ring,
                               ntime_gulp=args.gulp_size, core=cores.pop(0), gpu=gpus.pop(0)))
    ops.append(TEngineOp(log, tengine_ring, write_ring,
                         ntime_gulp=args.gulp_size*4096//1960, core=cores.pop(0), gpu=gpus.pop(0)))
    ops.append(WriterOp(log, write_ring,
                        npkt_gulp=32, core=cores.pop(0)))
    #ops.append(GlobalLogger(log, args, FILE_QUEUE))
    #ops.append(VoltageBeamCommandProcessor(log, args.record_directory, FILE_QUEUE, DRX_QUEUE))
    
    """
    t_now = LWATime(datetime.utcnow() + timedelta(seconds=15), format='datetime', scale='utc')
    mjd_now = int(t_now.mjd)
    mpm_now = int((t_now.mjd - mjd_now)*86400.0*1000.0)
    ops[-1].record(json.dumps({'id': 234343423,
                               'start_mjd': mjd_now,
                               'start_mpm': mpm_now,
                               'duration_ms': 30*1000}))
    
    try:
        os.unlink(FILE_QUEUE[0].filename)
    except OSError:
        pass
    """
    
    # Setup the threads
    threads = [threading.Thread(target=op.main) for op in ops]
    
    # Setup signal handling
    shutdown_event = setup_signal_handling(ops)
    ops[0].shutdown_event = shutdown_event
    #ops[-2].shutdown_event = shutdown_event
    #ops[-1].shutdown_event = shutdown_event
    
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
    
