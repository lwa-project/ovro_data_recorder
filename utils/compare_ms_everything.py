#!/usr/bin/env python3

import os
import sys
import numpy

from lsl.common.mcs import mjdmpm_to_datetime
from lsl.imaging.utils import CorrelatedDataMS

def main(args):
    # Sort out the filenames
    npfile, msfile = args[:2]
    if npfile[-4:] != '.npy':
        npfile, msfile = msfile, npfile
    assert(npfile[-4:] == '.npy')
    assert((msfile[-4:] == '.tar') or (msfile[-3:] == '.ms'))
    print(f"Comparing {os.path.basename(npfile)} with {os.path.basename(msfile)}")
    
    # Make sure we have simultaneous data
    np_mjd, np_hms, _ = os.path.basename(npfile).split('_', 2)
    np_ymd = mjdmpm_to_datetime(int(np_mjd, 10), 0).strftime("%Y%m%d")
    ms_ymd, ms_hms, _ = os.path.basename(msfile).split('_', 2)
    assert(np_ymd == ms_ymd)
    assert(np_hms == ms_hms)
    
    # Load in the "everything" file that contains, well, everything
    everything = numpy.load(npfile)
    everything = everything[...,0] + 1j*everything[...,1]
    everything = everything.astype(numpy.complex64)
    
    # Parse out the shape to figure out what we have
    nint, nbl, nchan, npol = everything.shape
    nant = int(numpy.sqrt(nbl*2))
    assert(nant*(nant+1)//2 == nbl)
    
    # CorrelatedDataMS only returns the baselines and not the auto-correlations.
    # Figure out where those are in "everything"
    bl = []
    k = 0
    for i in range(nant):
        for j in range(i, nant):
            if i != j:
                bl.append(k)
            k += 1
            
    # Load in the measurment set
    ms = CorrelatedDataMS(msfile)
    
    # Loop over integrations
    for i in range(nint):
        ## Everything's bit
        e = everything[i,...]
        
        ## Measurement set's bit - converted to a numpy array
        m = ms.get_data_set(i)
        m = [getattr(m, p, None).data for p in ('XX', 'XY', 'YX', 'YY')]
        m = numpy.array(m)
        m = m.transpose(1,2,0)
        
        # Compare
        print(f"  Integration {i+1}:")
        for p,pol in enumerate(('XX', 'XY', 'YX', 'YY')):
            diff = e[bl,:,p] - m[:,:,p]
            print(f"    {pol} -> min={diff.min()}, mean={diff.mean()}, max={diff.max()}")


if __name__ == '__main__':
    main(sys.argv[1:])
