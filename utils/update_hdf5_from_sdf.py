#!/usr/bin/env python3

import os
import sys
import h5py

from data import fill_from_sdf

h = h5py.File(sys.argv[1], mode='a')
s = sys.argv[2]

fill_from_sdf(h, s, station='ovrolwa')
h.close()
