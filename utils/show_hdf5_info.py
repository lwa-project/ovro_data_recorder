#!/usr/bin/env python3

import os
import sys
import h5py
import numpy
from datetime import datetime


def main(args):
    for filename in args:
        f = h5py.File(filename)
        time = f['/Observation1/time'][...]
        valid = numpy.where(time.getfield('int') > 0)[0]
        tstart = time[valid[0]]
        tint = time[valid[1]][0] - time[valid[0]][0]
        tint = tint + time[valid[1]][1] - time[valid[0]][1]
        tstop = time[valid[-1]]
        unfilled = time.size-len(valid)
        print('Time Range:')
        print('  Start:', tstart, '->', datetime.utcfromtimestamp(tstart[0]+tstart[1]))
        print('  Stop: ', tstop, '->', datetime.utcfromtimestamp(tstop[0]+tstop[1]))
        print('  Integration time:', tint, 's')
        print('  Unfilled integrations:', unfilled, '(%.3f s)' % (tint*unfilled,))
        
        freq = f['/Observation1/Tuning1/freq'][...]
        print('Frequency Range:')
        print('  Start:', '%.3f MHz' % (freq[0]/1e6,))
        print('  Stop: ', '%.3f MHz' % (freq[-1]/1e6,))
        print('  Channel width: %.3f kHz' % ((freq[1]-freq[0])/1e3,))
        
        tuning = f.get('/Observation1/Tuning1', None)
        print("Data Sets:")
        for key in tuning.keys():
            if key == 'freq':
                continue
            print(' ', key, '@', tuning[key].shape, 'with', tuning[key].dtype)
            
        f.close()


if __name__ == "__main__":
    main(sys.argv[1:])
