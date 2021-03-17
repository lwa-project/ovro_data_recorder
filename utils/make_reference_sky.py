#!/usr/bin/env python3

import os
import sys
import numpy
import shutil

from astropy.time import Time as AstroTime

from lsl.common import stations
from lsl.sim import vis as simVis

# Crude but effective for making sure we are up-to-date
try:
    os.unlink('station.py')
    os.unlink('ovro.txt')
except OSError:
    pass
shutil.copy('../station.py', 'station.py')
shutil.copy('../ovro.txt', 'ovro.txt')
from station import ovro


# Convert a station.Station to a lsl.common.stations.LWAStation object
ovro2 = stations.LWAStation('OVRO', ovro.lat*180/numpy.pi, ovro.lon*180/numpy.pi, ovro.elev, 'OV')
for ant in ovro.antennas:
    enz = ovro2.get_enz_offset((ant.lat*180/numpy.pi, ant.lon*180/numpy.pi, ant.elev))
    s = stations.Stand(ant.id, *enz)
    for pol in (0, 1):
        c = stations.Cable(f"Cable{ant.id:03d}-Pol{pol}", 0.0)
        a = stations.Antenna(ant.id*2+pol, stand=s, cable=c, pol=pol)
        ovro2.antennas.append(a)
ovro = ovro2

# Simulation setup
chan0 = 1234
nchan = 184
CHAN_BW = 196e6 / 8192
jd = AstroTime.now().jd

# Simulation array
freqs = (chan0 + numpy.arange(nchan)) * CHAN_BW + CHAN_BW/2
aa = simVis.build_sim_array(ovro, ovro.antennas[0::2], freqs/1e9, jd=jd)

# Simulate with bright sources only
dataSet = simVis.build_sim_data(aa, simVis.SOURCES, pols=['xx','yy'], jd=jd)
nbl, _ = dataSet.XX.data.shape

# Make the final data set that can be used with dr_visibilities.py
# NOTE:  XY and XY are 1% of XX and have sign flip between XY and YX
vis = numpy.zeros((nbl,nchan,4), dtype=numpy.complex64)
vis[:,:,0] = dataSet.XX.data
vis[:,:,1] = dataSet.XX.data* 0.01
vis[:,:,2] = dataSet.XX.data*-0.01
vis[:,:,3] = dataSet.YY.data

# Done
numpy.save('sky.npy', vis)
