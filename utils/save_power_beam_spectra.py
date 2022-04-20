#!/usr/bin/env python3

import os
import sys
import argparse
from datetime import datetime

from mnc.mcs import ImageMonitorPoint, Client

def main(args):
    c = Client()
    s = c.read_monitor_point('diagnostics/spectra', 'dr%s' % args.beam)
    t = datetime.utcfromtimestamp(s.timestamp)
    s = ImageMonitorPoint(s)
    s.to_file(args.output)
    print(f"Saved spectra from power beam {args.beam} at {t} to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description='Simple utility to poll the "diagnostics/spectra" monitoring point for a power beam pipeline and save it to disk',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument('-b', '--beam', type=int, default=1,
                        help='beam to query')
    parser.add_argument('-o', '--output', type=str, default='spectra.png',
                        help='filename to save the spectra to')
    args = parser.parse_args()
    main(args)
