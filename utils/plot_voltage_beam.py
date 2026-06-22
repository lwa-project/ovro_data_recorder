#!/usr/bin/env python3

import os
import sys
import struct
import numpy as np
from datetime import datetime

from matplotlib import pyplot as plt


"""
RBeam Packet Headers:
    uint8_t  server;   // Note: 1-based
    uint8_t  gbe;      // (AKA tuning)
    uint16_t nchan;    // Note: Big endian; 109
    uint8_t  nbeam;    // 2
    uint8_t  nserver;  // 6
    // Note: Big endian
    uint16_t chan0;    // First chan in packet
    uint64_t seq;      // Note: 1-based

Note:
    time_tag = seq * 2*4096
    time_tag is in units of ticks of a 196 MHz clock since the UNIX epoch
"""


def main(args):
    for filename in args:
        prev_timetag = 0
        nframe = 0
        with open(filename, 'rb') as fh:
            spec = []
            while True:
                hdr = fh.read(16)
                if len(hdr) < 16:
                    break
                    
                hdr = struct.unpack('>BBHBBHQ', hdr)
                if prev_timetag == 0:
                    print("Setup:")
                    print(f"  nchan: {hdr[2]} = {hdr[2]*196e6/(2*4096)/1e6:.3f} MHz")
                    print(f"  nbeam: {hdr[3]}")
                    print(f"  chan0: {hdr[-2]} = {hdr[-2]*196e6/(2*4096)/1e6:.3f} MHz")
                    print(f"  seq:   {hdr[-1]} = {datetime.utcfromtimestamp(hdr[-1]*2*4096/196e6)}")
                    nbeam = hdr[3]
                    nchan = hdr[2]
                    data_size = nbeam*nchan*2*8
                    chan0 = hdr[-2]
                else:
                    if hdr[2] != nchan or hdr[3] != nbeam:
                        print("Setup Change:")
                        print(f"  nchan: {hdr[2]} = {hdr[2]*196e6/(2*4096)/1e6:.3f} MHz")
                        print(f"  nbeam: {hdr[3]}")
                        print(f"  chan0: {hdr[-2]} = {hdr[-2]*196e6/(2*4096)/1e6:.3f} MHz")
                        print(f"  seq:   {hdr[-1]} = {datetime.utcfromtimestamp(hdr[-1]*2*4096/196e6)}")
                        nbeam = hdr[3]
                        nchan = hdr[2]
                        data_size = nbeam*nchan*2*8
                        chan0 = hdr[-2]
                        
                    if hdr[-1] - prev_timetag != 1:
                        timetag = hdr[-1] * 2 * 4096 / 196e6
                        print(f"WARNING: time tag skipped by {hdr[-1] - prev_timetag} sequencies at {datetime.utcfromtimestamp(timetag)}")
                        print(f"  {prev_timetag} -> {hdr[-1]}")
                prev_timetag = hdr[-1]
                
                data = fh.read(data_size)
                if len(data) < data_size:
                    break
                data = np.frombuffer(data, dtype=np.complex64)
                data = data.reshape(nbeam,nchan,2)
                
                spec.append(np.abs(data)**2)
                
                nframe += 1
                
            print(f"Frames Read: {nframe} = {nframe * 2 * 4096 / 196e6:.3f} s")
            
            spec = np.array(spec)
            int_spec = spec.mean(axis=0)
            
            chans = chan0 + np.arange(nchan)
            freqs = chans * 196e6/8192
            
            fig = plt.figure()
            gs = fig.add_gridspec(2, 2, hspace=0)
            (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col')
            ax3.plot(freqs/1e6, np.log10(int_spec[0,:,0])*10)
            ax4.plot(freqs/1e6, np.log10(int_spec[0,:,1])*10)
            
            ax1.imshow(np.log10(spec[:,0,:,0])*10, extent=(freqs[0]/1e6, freqs[-1]/1e6, 0, spec.shape[0]), vmin=ax3.get_ylim()[0], vmax=ax3.get_ylim()[1])
            ax1.axis('auto')
            ax2.imshow(np.log10(spec[:,0,:,1])*10, extent=(freqs[0]/1e6, freqs[-1]/1e6, 0, spec.shape[0]), vmin=ax4.get_ylim()[0], vmax=ax4.get_ylim()[1])
            ax2.axis('auto')
            
            ax3.set_xlabel('Frequency [MHz]')
            ax4.set_xlabel('Frequency [MHz]')
            ax3.set_ylabel('PSD [dB]')
            ax1.set_ylabel('Spectrum #')
            
            ax1.set_title('XX')
            ax2.set_title('YY')
            
            fig.suptitle(os.path.basename(filename))
            plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
