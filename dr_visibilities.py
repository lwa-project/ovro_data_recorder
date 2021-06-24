#!/usr/bin/env python

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
from station import ovro
from reductions import *
from operations import FileOperationsQueue
from monitoring import GlobalLogger
from control import VisibilityCommandProcessor
from lwams import get_zenith_uvw
from mcs import ImageMonitorPoint, MultiMonitorPoint, Client

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


import PIL.Image, PIL.ImageDraw, PIL.ImageFont


BASE_PATH = os.path.dirname(os.path.abspath(__file__))


QUEUE = FileOperationsQueue()


class CaptureOp(object):
    def __init__(self, log, sock, oring, nbl, ntime_gulp=1,
                 slot_ntime=6, fast=False, shutdown_event=None, core=None):
        self.log     = log
        self.sock    = sock
        self.oring   = oring
        self.nbl     = nbl
        self.ntime_gulp = ntime_gulp
        self.slot_ntime = slot_ntime
        self.fast    = fast
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        self.core    = core
        
    def shutdown(self):
        self.shutdown_event.set()
        
    def seq_callback(self, seq0, time_tag, chan0, nchan, navg, nsrc, hdr_ptr, hdr_size_ptr):
        print("++++++++++++++++ seq0     =", seq0)
        print("                 time_tag =", time_tag)
        hdr = {'time_tag': time_tag,
               'seq0':     seq0, 
               'chan0':    chan0,
               'cfreq':    chan0*CHAN_BW,
               'nchan':    nchan,
               'bw':       nchan*CHAN_BW*(4 if self.fast else 1),
               'navg':     navg,
               'nstand':   int(numpy.sqrt(8*nsrc+1)-1)//2,
               'npol':     2,
               'nbl':      nsrc,
               'complex':  True,
               'nbit':     32}
        print("******** CFREQ:", hdr['cfreq'])
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
        seq_callback.set_cor(self.seq_callback)
        
        with UDPCapture("cor", self.sock, self.oring, self.nbl, 0, 9000, 
                        self.ntime_gulp, self.slot_ntime,
                        sequence_callback=seq_callback, core=self.core) as capture:
            while not self.shutdown_event.is_set():
                status = capture.recv()
                if status in (1,4,5,6):
                    break
        del capture


class ReaderOp(object):
    def __init__(self, log, filename, oring, nbl, ntime_gulp=1,
                 slot_ntime=6, fast=False, shutdown_event=None, core=None):
        self.log      = log
        self.filename = filename
        self.oring    = oring
        self.nbl      = nbl
        self.ntime_gulp = ntime_gulp
        self.slot_ntime = slot_ntime
        self.fast     = fast
        if shutdown_event is None:
            shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event
        self.core     = core
        
    def shutdown(self):
        self.shutdown_event.set()

    def seq_callback(self, seq0, time_tag, chan0, nchan, navg, nsrc, hdr_ptr, hdr_size_ptr):
        print("++++++++++++++++ seq0     =", seq0)
        print("                 time_tag =", time_tag)
        hdr = {'time_tag': time_tag,
               'seq0':     seq0, 
               'chan0':    chan0,
               'cfreq':    chan0*CHAN_BW,
               'nchan':    nchan,
               'bw':       nchan*CHAN_BW*(4 if self.fast else 1),
               'navg':     navg,
               'nstand':   int(numpy.sqrt(8*nsrc+1)-1)//2,
               'npol':     2,
               'nbl':      nsrc,
               'complex':  True,
               'nbit':     32}
        print("******** CFREQ:", hdr['cfreq'])
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
        seq_callback.set_cor(self.seq_callback)
        
        navg  = 2400 if self.fast else 240000
        tint  = navg / CHAN_BW
        tgulp = tint * self.ntime_gulp
        
        with open(self.filename, 'rb') as fh:
            with DiskReader("cor_%i" % (192//4 if self.fast else 192), fh, self.oring, self.nbl, 0,  
                            self.ntime_gulp, self.slot_ntime,
                            sequence_callback=seq_callback, core=self.core) as capture:
                prev_time = time.time()
                while not self.shutdown_event.is_set():
                    status = capture.recv()
                    if status in (1,4,5,6):
                        break
                        
                    curr_time = time.time()
                    while curr_time - prev_time < tgulp:
                        time.sleep(0.01)
                        curr_time = time.time()
                    prev_time = curr_time


class DummyOp(object):
    def __init__(self, log, sock, oring, nbl, ntime_gulp=1,
                 slot_ntime=6, fast=False, shutdown_event=None, core=None):
        self.log     = log
        self.sock    = sock
        self.oring   = oring
        self.nbl     = nbl
        self.ntime_gulp = ntime_gulp
        self.slot_ntime = slot_ntime
        self.fast    = fast
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
            navg  = 2400 if self.fast else 240000
            tint  = navg / CHAN_BW
            tgulp = tint * self.ntime_gulp
            nsrc  = self.nbl
            nbl   = self.nbl
            chan0 = 1234
            nchan = 192 // (4 if self.fast else 1)
            npol = 4
            
            # Try to load model visibilities
            try:
                vis_base = numpy.load('utils/sky.npy')
            except:
                self.log.warn("Could not load model visibilities from utils/sky.py, using random data")
                vis_base = numpy.zeros((nbl, nchan, npol), dtype=numpy.complex64)
            assert(vis_base.shape[0] >= nbl)
            assert(vis_base.shape[1] >= nchan)
            assert(vis_base.shape[2] == npol)
            
            vis_base = vis_base[:self.nbl,::(4 if self.fast else 1),:]
            
            ohdr = {'time_tag': int(int(time.time())*FS),
                    'seq0':     0, 
                    'chan0':    chan0,
                    'cfreq':    chan0*CHAN_BW,
                    'nchan':    nchan,
                    'bw':       nchan*CHAN_BW*(4 if self.fast else 1),
                    'navg':     navg,
                    'nstand':   int(numpy.sqrt(8*nsrc+1)-1)//2,
                    'npol':     npol,
                    'nbl':      nbl,
                    'complex':  True,
                    'nbit':     32}
            ohdr_str = json.dumps(ohdr)
            
            ogulp_size = self.ntime_gulp*nbl*nchan*npol*8      # complex64
            oshape = (self.ntime_gulp,nbl,nchan,npol)
            self.oring.resize(ogulp_size)
            
            prev_time = time.time()
            with oring.begin_sequence(time_tag=ohdr['time_tag'], header=ohdr_str) as oseq:
                while not self.shutdown_event.is_set():
                    with oseq.reserve(ogulp_size) as ospan:
                        curr_time = time.time()
                        reserve_time = curr_time - prev_time
                        prev_time = curr_time
                        
                        odata = ospan.data_view(numpy.complex64).reshape(oshape)
                        odata[...] = vis_base + 0.01*numpy.random.randn(*oshape)
                        
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
    def __init__(self, log, id, iring, ntime_gulp=1, guarantee=True, core=-1):
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
        
        self.in_proclog.update({'nring':1, 'ring0':self.iring.name})
        
    def _plot_spectra(self, time_tag, freq, specs):
        # Plotting setup
        nchan = freq.size
        nstand = specs.shape[0]
        try:
            minval = numpy.min(specs[numpy.where(numpy.isfinite(specs))])
            maxval = numpy.max(specs[numpy.where(numpy.isfinite(specs))])
        except ValueError:
            minval = 0.0
            maxval = 1.0
            
        # Image setup
        width = 20
        height = 18
        im = PIL.Image.new('RGB', (width * 65 + 1, height * 65 + 21), '#FFFFFF')
        draw = PIL.ImageDraw.Draw(im)
        font = PIL.ImageFont.load(os.path.join(BASE_PATH, 'fonts', 'helvB10.pil'))
       
        # Axes boxes
        for i in range(width + 1):
            draw.line([i * 65, 0, i * 65, height * 65], fill = '#000000')
        for i in range(height + 1):
            draw.line([(0, i * 65), (im.size[0], i * 65)], fill = '#000000')
            
        # Power as a function of frequency for all antennas
        x = numpy.arange(nchan) * 64 // nchan
        for s in range(nstand):
            if s >= height * width:
                break
            x0, y0 = (s % width) * 65 + 1, (s // width + 1) * 65
            draw.text((x0 + 5, y0 - 60), str(s+1), font=font, fill='#000000')
            
            ## XX
            c = '#1F77B4'
            y = ((54.0 / (maxval - minval)) * (specs[s,:,0] - minval)).clip(0, 54)
            draw.point(list(zip(x0 + x, y0 - y)), fill=c)
            
            ## YY
            c = '#FF7F0E'
            y = ((54.0 / (maxval - minval)) * (specs[s,:,1] - minval)).clip(0, 54)
            draw.point(list(zip(x0 + x, y0 - y)), fill=c)
            
        # Summary
        ySummary = height * 65 + 2
        timeStr = datetime.utcfromtimestamp(time_tag / FS)
        timeStr = timeStr.strftime("%Y/%m/%d %H:%M:%S UTC")
        draw.text((5, ySummary), timeStr, font = font, fill = '#000000')
        rangeStr = 'range shown: %.3f to %.3f dB' % (minval, maxval)
        draw.text((210, ySummary), rangeStr, font = font, fill = '#000000')
        x = im.size[0] + 15
        for label, c in reversed(list(zip(('XX',     'YY'),
                                          ('#1F77B4','#FF7F0E')))):
            x -= draw.textsize(label, font = font)[0] + 20
            draw.text((x, ySummary), label, font = font, fill = c)
            
        return im
        
    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        for iseq in self.iring.read(guarantee=self.guarantee):
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            self.log.info("Spectra: Start of new sequence: %s", str(ihdr))
            
            # Setup the ring metadata and gulp sizes
            time_tag = ihdr['time_tag']
            navg     = ihdr['navg']
            nbl      = ihdr['nbl']
            nstand   = ihdr['nstand']
            chan0    = ihdr['chan0']
            nchan    = ihdr['nchan']
            chan_bw  = ihdr['bw'] / nchan
            npol     = ihdr['npol']
            
            igulp_size = self.ntime_gulp*nbl*nchan*npol*8   # complex64
            ishape = (self.ntime_gulp,nbl,nchan,npol)
            
            # Setup the arrays for the frequencies and auto-correlations
            freq = chan0*chan_bw + numpy.arange(nchan)*chan_bw
            autos = [i*(2*(nstand-1)+1-i)//2 + i for i in range(nstand)]
            last_save = 0.0
            
            prev_time = time.time()
            for ispan in iseq.read(igulp_size):
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                ## Setup and load
                idata = ispan.data_view(numpy.complex64).reshape(ishape)
                
                if time.time() - last_save > 60:
                    ## Timestamp
                    tt = LWATime(time_tag, format='timetag')
                    ts = tt.unix
                    
                    ## Pull out the auto-correlations
                    adata = idata[0,autos,:,:].real
                    adata = adata[:,:,[0,3]]
                    
                    ## Plot
                    im = self._plot_spectra(time_tag, freq, 10*numpy.log10(adata))
                    
                    ## Save
                    mp = ImageMonitorPoint.from_image(im)
                    self.client.write_monitor_point('diagnostics/spectra',
                                                    mp, timestamp=ts)
                    
                    if True:
                        ## Save again, this time to disk
                        mjd, dt = tt.mjd, tt.datetime
                        mjd = int(mjd)
                        h, m, s = dt.hour, dt.minute, dt.second
                        filename = '%06i_%02i%02i%02i_spectra.png' % (mjd, h, m, s)
                        mp.to_file(filename)
                        
                    last_save = time.time()
                    
                time_tag += navg * self.ntime_gulp * (int(FS) // int(CHAN_BW))
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': 0.0, 
                                          'process_time': process_time,})
                
        self.log.info("SpectraOp - Done")


class BaselineOp(object):
    def __init__(self, log, id, station, iring, ntime_gulp=1, guarantee=True, core=-1):
        self.log        = log
        self.station    = station
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
        
        self.in_proclog.update({'nring':1, 'ring0':self.iring.name})
        
    def _plot_baselines(self, time_tag, freq, dist, baselines, valid):
        # Plotting setup
        nchan = freq.size
        nbl = baselines.shape[0]
        freq = freq[nchan//2]
        baselines = baselines[valid,nchan//2,:]
        baselines = numpy.abs(baselines[:,[0,1,3]])
        minval = numpy.min(baselines)
        maxval = numpy.max(baselines)
        if minval == maxval:
            maxval = minval + 1.0
            
        mindst = 0.0
        maxdst = numpy.max(dist)
        
        # Image setup
        im = PIL.Image.new('RGB', (601, 421), '#FFFFFF')
        draw = PIL.ImageDraw.Draw(im)
        font = PIL.ImageFont.load(os.path.join(BASE_PATH, 'fonts', 'helvB10.pil'))
        
        # Axes boxes
        for i in range(2):
            draw.line([i * 600, 0, i * 600, 400], fill = '#000000')
        for i in range(2):
            draw.line([(0, i * 400), (im.size[0], i * 400)], fill = '#000000')
            
        # Visiblity amplitudes as a function of (u,v) distance
        x0, y0 = 1, 400
        draw.text((x0 + 500, y0 - 395), '%.3f MHz' % (freq/1e6,), font=font, fill='#000000')
        
        ## (u,v) distance
        x = ((599.0 / (maxdst - mindst)) * (dist - mindst)).clip(0, 599)
        
        ## XX
        y = ((399.0 / (maxval - minval)) * (baselines[:,0] - minval)).clip(0, 399)
        draw.point(list(zip(x0 + x, y0 - y)), fill='#1F77B4')
        
        ## YY
        y = ((399.0 / (maxval - minval)) * (baselines[:,2] - minval)).clip(0, 399)
        draw.point(list(zip(x0 + x, y0 - y)), fill='#FF7F0E')
        
        ### XY
        #y = ((399.0 / (maxval - minval)) * (baselines[:,1] - minval)).clip(0, 399)
        #draw.point(list(zip(x0 + x, y0 - y)), fill='#A00000')
        
        # Details and labels
        ySummary = 402
        timeStr = datetime.utcfromtimestamp(time_tag / FS)
        timeStr = timeStr.strftime("%Y/%m/%d %H:%M:%S UTC")
        draw.text((5, ySummary), timeStr, font = font, fill = '#000000')
        rangeStr = 'range shown: %.6f - %.6f' % (minval, maxval)
        draw.text((210, ySummary), rangeStr, font = font, fill = '#000000')
        x = im.size[0] + 15
        #for label, c in reversed(list(zip(('XX','XY','YY'), ('#1F77B4','#A00000','#FF7F0E')))):
        for label, c in reversed(list(zip(('XX','YY'), ('#1F77B4','#FF7F0E')))):
            x -= draw.textsize(label, font = font)[0] + 20
            draw.text((x, ySummary), label, font = font, fill = c)
            
        return im
        
    def main(self):
        cpu_affinity.set_core(self.core)
        self.bind_proclog.update({'ncore': 1, 
                                  'core0': cpu_affinity.get_core(),})
        
        for iseq in self.iring.read(guarantee=self.guarantee):
            ihdr = json.loads(iseq.header.tostring())
            
            self.sequence_proclog.update(ihdr)
            
            self.log.info("Baseline: Start of new sequence: %s", str(ihdr))
            
            # Setup the ring metadata and gulp sizes
            time_tag = ihdr['time_tag']
            navg     = ihdr['navg']
            nbl      = ihdr['nbl']
            nstand   = ihdr['nstand']
            chan0    = ihdr['chan0']
            nchan    = ihdr['nchan']
            chan_bw  = ihdr['bw'] / nchan
            npol     = ihdr['npol']
            
            igulp_size = self.ntime_gulp*nbl*nchan*npol*8
            ishape = (self.ntime_gulp,nbl,nchan,npol)
            self.iring.resize(igulp_size)
            
            # Setup the arrays for the frequencies and baseline lenghts
            freq = chan0*chan_bw + numpy.arange(nchan)*chan_bw
            uvw = get_zenith_uvw(self.station, LWATime(time_tag, format='timetag'))
            uvw[:,2] = 0
            dist = numpy.sqrt((uvw**2).sum(axis=1))
            valid = numpy.where(dist > 0.1)[0]
            last_save = 0.0
            
            prev_time = time.time()
            for ispan in iseq.read(igulp_size):
                if ispan.size < igulp_size:
                    continue # Ignore final gulp
                curr_time = time.time()
                acquire_time = curr_time - prev_time
                prev_time = curr_time
                
                ## Setup and load
                idata = ispan.data_view(numpy.complex64).reshape(ishape)
                
                if time.time() - last_save > 60:
                    ## Timestamp
                    tt = LWATime(time_tag, format='timetag')
                    ts = tt.unix
                    
                    ## Plot
                    im = self._plot_baselines(time_tag, freq, dist, idata[0,...], valid)
                    
                    ## Save
                    mp = ImageMonitorPoint.from_image(im)
                    self.client.write_monitor_point('diagnostics/baselines',
                                                    mp, timestamp=ts)
                    
                    if True:
                        ## Save again, this time to disk
                        mjd, dt = tt.mjd, tt.datetime
                        mjd = int(mjd)
                        h, m, s = dt.hour, dt.minute, dt.second
                        filename = '%06i_%02i%02i%02i_baselines.png' % (mjd, h, m, s)
                        mp.to_file(filename)
                        
                    last_save = time.time()
                    
                time_tag += navg * self.ntime_gulp * (int(FS) // int(CHAN_BW))
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': 0.0, 
                                          'process_time': process_time,})
                
        self.log.info("BaselineOp - Done")


class StatisticsOp(object):
    def __init__(self, log, id, iring, ntime_gulp=1, guarantee=True, core=None):
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
            nbl      = ihdr['nbl']
            nstand   = ihdr['nstand']
            chan0    = ihdr['chan0']
            nchan    = ihdr['nchan']
            chan_bw  = ihdr['bw'] / nchan
            npol     = ihdr['npol']
            
            igulp_size = self.ntime_gulp*nbl*nchan*npol*8        # complex64
            ishape = (self.ntime_gulp,nbl,nchan,npol)
            
            autos = [i*(2*(nstand-1)+1-i)//2 + i for i in range(nstand)]
            data_pols = ['XX', 'YY']
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
                idata = ispan.data_view(numpy.complex64).reshape(ishape)
                
                if time.time() - last_save > 60:
                    ## Timestamp
                    tt = LWATime(time_tag, format='timetag')
                    ts = tt.unix
                    
                    ## Pull out the auto-correlations
                    adata = idata[0,autos,:,:].real
                    adata = adata[:,:,[0,3]]
                    
                    ## Run the statistics over all times/channels
                    ##  * only really works for ntime_gulp=1
                    data_min = numpy.min(adata, axis=1)
                    data_max = numpy.max(adata, axis=1)
                    data_avg = numpy.mean(adata, axis=1)
                    
                    ## Save
                    for data,name in zip((data_min,data_avg,data_max), ('min','avg','max')):
                        value = MultiMonitorPoint([data[:,i].tolist() for i in range(data.shape[1])],
                                                  timestamp=ts, field=data_pols)
                        self.client.write_monitor_point('statistics/%s' % name, value)
                        
                    last_save = time.time()
                    
                time_tag += navg * self.ntime_gulp * (int(FS) // int(CHAN_BW))
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': -1, 
                                          'process_time': process_time,})
                
        self.log.info("StatisticsOp - Done")


class WriterOp(object):
    def __init__(self, log, station, iring, ntime_gulp=1, fast=False, guarantee=True, core=None):
        self.log        = log
        self.station    = station
        self.iring      = iring
        self.ntime_gulp = ntime_gulp
        self.fast       = fast
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
            
            # Setup the ring metadata and gulp sizes
            time_tag = ihdr['time_tag']
            navg     = ihdr['navg']
            nbl      = ihdr['nbl']
            chan0    = ihdr['chan0']
            nchan    = ihdr['nchan']
            chan_bw  = ihdr['bw'] / nchan
            npol     = ihdr['npol']
            pols     = ['XX','XY','YX','YY']
            
            igulp_size = self.ntime_gulp*nbl*nchan*npol*8        # complex64
            ishape = (self.ntime_gulp,nbl,nchan,npol)
            self.iring.resize(igulp_size, 10*igulp_size*(10 if self.fast else 1))
            
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
                
                ## On our first span, update the pipeline lag for the queue
                ## so that we start recording at the right times
                if first_gulp:
                    QUEUE.update_lag(LWATime(time_tag, format='timetag').datetime)
                    self.log.info("Current pipeline lag is %s", QUEUE.lag)
                    first_gulp = False
                    
                ## Setup and load
                idata = ispan.data_view(numpy.complex64).reshape(ishape)
               
                ## Determine what to do
                if QUEUE.active is not None:
                    ### Recording active - write
                    if not QUEUE.active.is_started:
                        self.log.info("Started operation - %s", QUEUE.active)
                        QUEUE.active.start(self.station, chan0, navg, nchan, chan_bw, npol, pols)
                        was_active = True
                    QUEUE.active.write(time_tag, idata)
                elif was_active:
                    ### Recording just finished
                    #### Clean
                    was_active = False
                    QUEUE.clean()
                    
                    #### Close
                    self.log.info("Ended operation - %s", QUEUE.previous)
                    QUEUE.previous.stop()
                    
                time_tag += navg * self.ntime_gulp * (int(FS) // int(CHAN_BW))
                
                curr_time = time.time()
                process_time = curr_time - prev_time
                prev_time = curr_time
                self.perf_proclog.update({'acquire_time': acquire_time, 
                                          'reserve_time': -1, 
                                          'process_time': process_time,})
                
        self.log.info("WriterOp - Done")


def main(argv):
    global QUEUE
    
    parser = argparse.ArgumentParser(
                 description="Data recorder for slow/fast visibility data"
                 )
    parser.add_argument('-a', '--address', type=str, default='127.0.0.1',
                        help='IP address to listen to')
    parser.add_argument('-p', '--port', type=int, default=10000,
                        help='UDP port to receive data on')
    parser.add_argument('-o', '--offline', action='store_true',
                        help='run in offline using the specified file to read from')
    parser.add_argument('--filename', type=str,
                        help='filename containing packets to read from in offline mode')
    parser.add_argument('-c', '--cores', type=str, default='0,1,2,3,4',
                        help='comma separated list of cores to bind to')
    parser.add_argument('-g', '--gulp-size', type=int, default=1,
                        help='gulp size for ring buffers')
    parser.add_argument('-l', '--logfile', type=str,
                        help='file to write logging to')
    parser.add_argument('-r', '--record-directory', type=str, default=os.path.abspath('.'),
                        help='directory to save recorded files to')
    parser.add_argument('-t', '--record-directory-quota', type=int, default=0,
                        help='quota for the recording directory, 0 disables the quota')
    parser.add_argument('-q', '--quick', action='store_true',
                        help='run in fast visibiltiy mode')
    parser.add_argument('-i', '--nint-per-file', type=int, default=1,
                        help='number of integrations to write per measurement set')
    parser.add_argument('-n', '--no-tar', action='store_true',
                        help='do not store the measurement sets inside a tar file')
    parser.add_argument('-f', '--fork', action='store_true',
                        help='fork and run in the background')
    args = parser.parse_args()
    
    # Process the -q/--quick option
    station = ovro
    if args.quick:
        args.nint_per_file = max([10, args.nint_per_file])
        station = ovro.select_subset(list(range(1, 48+1)))
        
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
        
    # Setup the subsystem ID
    mcs_id = 'drv'
    if args.quick:
        mcs_id += 'f'
    else:
        mcs_id += 's'
    mcs_id += args.address.split('.')[-1]
    
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
    
    # Setup antennas
    nant = len(station.antennas)
    nbl = nant*(nant+1)//2
    
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
        if args.filename:
            ops.append(ReaderOp(log, args.filename, capture_ring, nbl,
                                ntime_gulp=args.gulp_size, slot_ntime=6, fast=args.quick,
                                core=cores.pop(0)))
        else:
            ops.append(DummyOp(log, isock, capture_ring, nbl,
                               ntime_gulp=args.gulp_size, slot_ntime=6, fast=args.quick,
                               core=cores.pop(0)))
    else:
        ops.append(CaptureOp(log, isock, capture_ring, nbl,
                             ntime_gulp=args.gulp_size, slot_ntime=6, fast=args.quick,
                             core=cores.pop(0)))
    if not args.quick:
        ops.append(SpectraOp(log, mcs_id, capture_ring,
                             ntime_gulp=args.gulp_size, core=cores.pop(0)))
        ops.append(BaselineOp(log, mcs_id, station, capture_ring,
                              ntime_gulp=args.gulp_size, core=cores.pop(0)))
    ops.append(StatisticsOp(log, mcs_id, capture_ring,
                            ntime_gulp=args.gulp_size, core=cores.pop(0)))
    ops.append(WriterOp(log, station, capture_ring,
                        ntime_gulp=args.gulp_size, fast=args.quick, core=cores.pop(0)))
    ops.append(GlobalLogger(log, mcs_id, args, QUEUE, quota=args.record_directory_quota))
    ops.append(VisibilityCommandProcessor(log, mcs_id, args.record_directory, QUEUE,
                                          nint_per_file=args.nint_per_file,
                                          is_tarred=not args.no_tar))
    
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
        
    t_now = LWATime(datetime.utcnow() + timedelta(seconds=15), format='datetime', scale='utc')
    mjd_now = int(t_now.mjd)
    mpm_now = int((t_now.mjd - mjd_now)*86400.0*1000.0)
    c = Client()
    r = c.send_command(mcs_id, 'start',
                       start_mjd=mjd_now, start_mpm=mpm_now)
    print('III', r)
    
    t_now = LWATime(datetime.utcnow() + timedelta(seconds=75), format='datetime', scale='utc')
    mjd_now = int(t_now.mjd)
    mpm_now = int((t_now.mjd - mjd_now)*86400.0*1000.0)
    r = c.send_command(mcs_id, 'stop',
                       stop_mjd=mjd_now, stop_mpm=mpm_now)
    print('III', r)
    
    while not shutdown_event.is_set():
        signal.pause()
    log.info("Shutdown, waiting for threads to join")
    for thread in threads:
        thread.join()
    log.info("All done")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
    
