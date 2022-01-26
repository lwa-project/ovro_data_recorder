#!/usr/bin/env python3

import os
import sys
import h5py
import argparse

from data import fill_from_sdf

def main(args):
    print(f"Updating {os.path.basename(args.filename)} using {os.path.basename(args.sdfname)}")
    h = h5py.File(args.filename, mode='a')
    fill_from_sdf(h, args.sdfname, station='ovrolwa')
    h.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                description='Update the observation metadata in a HDF5 using the SDF that caused it to be recorded')
    parser.add_argument('filename', type=str,
                        help='HDF5 to update')
    parser.add_argument('sdfname', type=str,
                        help='associated SDF to use')
    args = parser.parse_args()
    main(args)
