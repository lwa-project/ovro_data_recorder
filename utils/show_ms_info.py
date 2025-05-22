#!/usr/bin/env python3

import os
import sys
import numpy
import shutil
import tempfile
import subprocess

from casacore.tables import table


def smart_int(value):
    for i in range(len(value)):
        try:
            return int(value[i:], 10)
        except ValueError:
            pass
    raise ValueError("Cannot find an integer in '%s'" % value)


def wrap_range_list(line, width=70, initial_indent='', subsequent_indent=''):
    parts = line.split(',')
    output = [initial_indent+parts[0]]
    for part in parts[1:]:
        part = part.strip().rstrip()
        if len(output[-1]) + len(part) < 70:
            output[-1] += ', '+part
        else:
            output.append(subsequent_indent)
            output[-1] += part
    return '\n'.join(output)


def build_metastack(flds, srcs, spws, dscs):
    return {'fields': flds,
            'sources': srcs,
            'windows': spws,
            'mapper': dscs}


def get_pointing(field_id, metastack):
    source_id =  metastack['fields'].getcol('SOURCE_ID')[field_id]
    direction = metastack['sources'].getcol('DIRECTION')[source_id]
    return direction


def get_frequencies(desc_id, metastack):
    window_id = metastack['mapper'].getcol('SPECTRAL_WINDOW_ID')[desc_id]
    freqs = metastack['windows'].getcol('CHAN_FREQ')[window_id]
    return freqs


def main(args):
    for filename in args:
        if not os.path.isdir(filename):
            if filename[-7:] == '.ms.tar':
                # Oh, it's tarred, extract it
                tempdir = tempfile.mkdtemp(prefix='casa_ms')
                subprocess.check_call(['tar', '-C', tempdir, '-x', '-f', filename])
                
                msname = os.path.basename(filename).replace('.tar', '')
                filename = os.path.join(tempdir, msname)
                
        data = table(filename, ack=False)
        ants = table(os.path.join(filename, 'ANTENNA'), ack=False)
        srcs = table(os.path.join(filename, 'SOURCE'), ack=False)
        flds = table(os.path.join(filename, 'FIELD'), ack=False)
        spws = table(os.path.join(filename, 'SPECTRAL_WINDOW'), ack=False)
        dscs = table(os.path.join(filename, 'DATA_DESCRIPTION'), ack=False)
        mstack = build_metastack(flds, srcs, spws, dscs)
        
        nvis = data.nrows()
        scans = numpy.unique(data.getcol('SCAN_NUMBER'))
        nscan = len(scans)
        nbl = nvis // nscan
        nant = ants.nrows()
        
        ant_ranges = []
        for i,name in enumerate(ants.getcol('NAME')):
            if i == 0:
                ant_ranges.append([name, name])
            else:
                if smart_int(ant_ranges[-1][1]) + 1 == smart_int(name):
                    ant_ranges[-1][1] = name
                else:
                    ant_ranges.append([name, name])
                    
        print("Data Sizes:")
        print("  Antenna count:", nant)
        # print("    Antenna Name Contiguous Ranges:")
        # print(wrap_range_list(', '.join(["%s to %s" % tuple(ant_range) for ant_range in ant_ranges]),
        #                       initial_indent='      ', subsequent_indent='      '))
        print("  Visibility count:", nvis)
        print("Scans:")
        for i,s in enumerate(scans):
            sdata = data.query(f"SCAN_NUMBER=={s}")
            
            stime = sdata.getcol('TIME') / 86400.0
            sintt = sdata.getcol('INTERVAL')
            print(f"  #{s}")
            print("    Time Range:")
            print("      Start:", min(stime), "@", len(numpy.where(stime == min(stime))[0]))
            print("      Stop:", max(stime), "@", len(numpy.where(stime == max(stime))[0]))
            print("      Integration time:", min(sintt), "to", max(sintt), "s")
            
            sdesc = sdata.getcol('DATA_DESC_ID')[0]
            freqs = get_frequencies(sdesc, mstack)
            print("    Frequency Range:")
            print("      Start:", freqs[0]/1e6, "MHz")
            print("      Stop:", freqs[-1]/1e6, "MHz")
            print("      Channel width:", ((freqs[1]-freqs[0])/1e3), "kHz")
            
            sfild = sdata.getcol('FIELD_ID')[0]
            dir = get_pointing(sfild, mstack)
            print("    Pointing:")
            print("      RA:", dir[0]*180/numpy.pi, "deg")
            print("      Dec:", dir[1]*180/numpy.pi, "deg")
            
            suvws = sdata.getcol('UVW')
            dist = numpy.sqrt(suvws[:,0]**2 + suvws[:,1]**2)
            nz_dist = dist[numpy.where(dist != 0.0)]
            print("     UVWs:")
            print("       Number at zero:", len(numpy.where(dist == 0.0)[0]), "of", dist.size)
            print("       (u,v) distance min:", min(nz_dist))
            print("       (u,v) distance max:", max(nz_dist))
            
            try:
                fill = data.getkeyword("FILL_LEVEL_%d" % i)
                print("     Packet Fill Level: %.3f%%" % (fill*100))
            except:
                pass
                
        data.close()
        ants.close()
        srcs.close()
        flds.close()
        spws.close()
        dscs.close()
        
        try:
            # Cleanup if we have extracted from a tar file
            shutil.rmtree(tempdir)
        except NameError:
            pass


if __name__ == '__main__':
    main(sys.argv[1:])
