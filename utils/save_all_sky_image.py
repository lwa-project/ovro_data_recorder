#!/usr/bin/env python3

import os
import sys
import argparse
from datetime import datetime, timedelta

from mnc.mcs import ImageMonitorPoint, Client

MAPPING = ['drvs7601', 'drvs7602',
           'drvs7701', 'drvs7702',
           'drvs7801', 'drvs7802',
           'drvs7901', 'drvs7902',
           'drvs8001', 'drvs8002',
           'drvs8101', 'drvs8102',
           'drvs8201', 'drvs8202',
           'drvs8301', 'drvs8302']

def main(args):
    c = Client()
    s = c.read_monitor_point('diagnostics/image', MAPPING[args.pipeline-1])
    t = datetime.utcfromtimestamp(s.timestamp)
    age = datetime.utcnow() - t
    if args.lwatv and age > timedelta(minutes=5):
        print(f"Image is {age.total_seconds()/60:.1f} minutes old, skipping")
        exit(1)
    s = ImageMonitorPoint(s)
    s.to_file(args.output)
    print(f"Saved all-sky image from {MAPPING[args.pipeline-1]} at {t} to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description='Simple utility to poll the "diagnostics/image" monitoring point for a slow visibility pipeline and save it to disk',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument('-p', '--pipeline', type=int, default=1,
                        help='one-based pipeline to query')
    parser.add_argument('-o', '--output', type=str, default='image.png',
                        help='filename to save the image to')
    parser.add_argument('--lwatv', action='store_true',
                        help='run in "LWATV" mode where the output image is only saved if it is recent (< 5 min)')
    args = parser.parse_args()
    main(args)
