#!/usr/bin/env python3

import os
import sys
import struct
import numpy
from datetime import datetime


"""
IBeam Packet Headers:
    uint8_t  server;   // Note: 1-based
    uint8_t  gbe;      // (AKA tuning)
    uint8_t  nchan;    // 109
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
            while True:
                hdr = fh.read(15)
                if len(hdr) < 15:
                    break
                    
                hdr = struct.unpack('>BBBBBHQ', hdr)
                if prev_timetag == 0:
                    print("Setup:")
                    print(f"  nchan: {hdr[2]} = {hdr[2]*196e6/(2*4096)/1e6:.3f} MHz")
                    print(f"  nbeam: {hdr[3]}")
                    print(f"  chan0: {hdr[-2]} = {hdr[-2]*196e6/(2*4096)/1e6:.3f} MHz")
                    print(f"  seq:   {hdr[-1]} = {datetime.utcfromtimestamp(hdr[-1]*2*4096/196e6)}")
                    nbeam = hdr[3]
                    nchan = hdr[2]
                    data_size = nbeam*nchan*2*8
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
                        
                    if hdr[-1] - prev_timetag != 1:
                        timetag = hdr[-1] * 2 * 4096 / 196e6
                        print(f"WARNING: time tag skipped by {hdr[-1] - prev_timetag} sequencies at {datetime.utcfromtimestamp(timetag)}")
                        print(f"  {prev_timetag} -> {hdr[-1]}")
                prev_timetag = hdr[-1]
                
                data = fh.read(data_size)
                if len(data) < data_size:
                    break
                #data = np.frombuffer(data, dtype=np.complex64)
                #data = data.reshape(nbeam,nchan,2)
                
                nframe += 1
                
            print(f"Frames Read: {nframe} = {nframe * 2 * 4096 / 196e6:.3f} s")


if __name__ == '__main__':
    main(sys.argv[1:])
