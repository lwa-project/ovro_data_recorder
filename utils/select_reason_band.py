#!/usr/bin/env python3

import numpy as np
import argparse

from mnc.xengine_beamformer_control import AllowedPipelineFailure, BeamPointingControl


#: REASON frequency of interest in Hz
REASON_FREQ = 60e6


def main(args):
    beam1 = BeamPointingControl(1)
    freqs = np.array(beam1.freqs)
    reason_band = np.argmin(np.abs(freqs - REASON_FREQ))
    reason_band //= freqs.shape[1]
    print(f"Found REASON frequency of interest in pipeline #{reason_band}")
    print(f"  This pipeline covers {beam1.freqs[reason_band][0]/1e6:.3f} to {beam1.freqs[reason_band][-1]/1e6:.3f} MHz")
    
    print("Reconfiguring x-engine voltage beam destination IPs/ports")
    for i,p in enumerate(beam1.pipelines):
        with AllowedPipelineFailure(p):
            if i == reason_band:
                p.beamform_vlbi_output.set_destination(args.dest_ip, args.dest_port)
            else:
                p.beamform_vlbi_output.set_destination('0.0.0.0', args.dest_port)
                
    print("To capture data with tcpdump:")
    if args.dest_ip == '10.41.0.97':
        print(f"  tcpdump -i ens30f1 -w reason.pcap udp dst port {args.dest_port}")
    else:
        print(f"  tcpdump -i <data_interface> -w reason.pcap udp dst port {args.dest_port}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to help control the voltage beam output so that only data containing the REASON frequency are sent',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--dest-ip', type=str, default='10.41.0.97',
                        help='destination IP address for the voltage beam data')
    parser.add_argument('--dest-port', type=int, default=21001,
                        help='destination UDP port for the voltage beam data')
    args = parser.parse_args()
    main(args)
