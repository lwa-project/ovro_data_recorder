#!/usr/bin/env python3

import os
import dpkt
import numpy as np
import struct
import argparse


"""
IBeam Packet Headers:
    uint8_t  server;   // Note: 1-based
    uint8_t  gbe;      // (AKA tuning)
    uint8_t  nchan;    // 109   -   should always be 96
    uint8_t  nbeam;    // 2
    uint8_t  nserver;  // 6
    // Note: Big endian
    uint16_t chan0;    // First chan in packet
    uint64_t seq;      // Note: 1-based -   *2*4096 to get in units of ticks of a196 MHz clock since the epoch
"""


def main(args):
    if args.outname is None:
        outname = os.path.basename(args.filename)
        outname, _ = os.path.splitext(outname)
        args.outname = outname+'.ibeam'
        
    with open(args.filename, 'rb') as fh:
        with open(args.outname, 'wb') as oh:
            for ts, pkt in dpkt.pcap.Reader(fh):
                eth = dpkt.ethernet.Ethernet(pkt)
                udp = eth.data.udp
                payload = udp.data
                
                oh.write(payload)
                
                #hdr = struct.unpack('>BBBBBHQ', payload[:15])
                #data = np.frombuffer(payload[15:], dtype=np.complex64)
                #data = data.reshape(-1,2)
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='read in a REASON pcap file and convert it into collection of raw ibeam packets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str,
                        help='filename to convert')
    parser.add_argument('-o', '--outname', type=str,
                        help='output filename; None = auto-name based on the input')
    args = parser.parse_args()
    main(args)
