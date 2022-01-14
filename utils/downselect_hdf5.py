#!/usr/bin/env python3

"""
Given an HDF5 file, decimate the data contained in it in both time and 
frequency, and save the results to a new file.
"""

import os
import sys
import h5py
import numpy
import argparse

def _fillHDF(inputH5, output, tDecimation=1, channels=None, level=0):
    """
    Function to recursively copy the structure of a HDF5 file created by 
    hdfWaterfall.py or drspec2hdf.py.
    """
    
    # Copy the attributes
    for key in inputH5.attrs:
        if key == 'tInt':
            value = inputH5.attrs[key]*tDecimation
        elif key == 'LFFT':
            value = len(channels)
        elif key == 'nChan':
            value = len(channels)
        else:
            value = inputH5.attrs[key]
        output.attrs[key] = value
        
    # Loop over the entities in the first input file for copying
    for ent in list(inputH5):
        ## Get the entity
        entity = inputH5.get(ent, None)
        print("%sWorking on '%s'..." % (' '*level*2, ent))
        
        ## Is it a group?
        if type(entity).__name__ == 'Group':
            ### If so, add it and fill it in.
            if ent not in list(output):
                entityO = output.create_group(ent)
            _fillHDF(entity, entityO, tDecimation=tDecimation, channels=channels, level=level+1)
            continue
            
        ## Is it a dataset?
        if type(entity).__name__ == 'Dataset':
            ### If so, add it and fill it in
            if ent in ('Steps', 'Delays', 'Gains'):
                entity0 = output.create_dataset(ent, entity.shape, entity.dtype)
                entity0[:] = entity[:]
                
            elif ent == 'time':
                newShape = (entity.shape[0]//tDecimation,)
                entityO = output.create_dataset(ent, newShape, entity.dtype)
                for i in range(newShape[0]):
                    data = entity[tDecimation*i:tDecimation*(i+1)]
                    entityO[i] = data[0]
                    
            elif ent == 'Saturation':
                newShape = (entity.shape[0]//tDecimation, entity.shape[1])
                entityO = output.create_dataset(ent, newShape, entity.dtype)
                for i in range(newShape[0]):
                    data = entity[tDecimation*i:tDecimation*(i+1),:]
                    entityO[i,:] = data.sum(axis=0)
                    
            elif ent == 'freq':
                newShape = (len(channels),)
                entityO = output.create_dataset(ent, newShape, entity.dtype)
                for i in range(newShape[0]):
                    data = entity[channels]
                    entityO[i] = data[i]
                    
            else:
                newShape = (entity.shape[0]//tDecimation, len(channels))
                entityO = output.create_dataset(ent, newShape, entity.dtype)
                for i in range(newShape[0]):
                    data = entity[tDecimation*i:tDecimation*(i+1),:]
                    data = data.mean(axis=0)
                    data = data[channels]
                    if data.dtype != entity.dtype:
                        data = data.astype(entity.dtype)
                    entityO[i,:] = data
                    
            ### Update the dataset attributes
            for key in entity.attrs:
                entityO.attrs[key] = entity.attrs[key]
                
    return True


def main(args):
    for filename in args.filename:
        outname = os.path.basename(filename)
        outname = os.path.splitext(outname)[0]
        outname = '%s-decim.hdf5' % outname
        
        if os.path.exists(outname):
            if not args.force:
                yn = input("WARNING: '%s' exists, overwrite? [Y/n] " % outname)
            else:
                yn = 'y'
                
            if yn not in ('n', 'N'):
                os.unlink(outname)
            else:
                print("WARNING: output file '%s' already exists, skipping" % outname)
                continue
                
        hIn  = h5py.File(filename, mode='r')
        hOut = h5py.File(outname, mode='w')
        
        # Figure out the valid channel range
        tuning = hIn.get('/Observation1/Tuning1', None)
        freq = tuning['freq'][:]
        nchan = freq.size
        if args.spectral_range is None:
            for key in tuning.keys():
                if key not in ('XX', 'I'):
                    continue
                nint, nchan = tuning[key].shape
                spec = tuning[key][nint//2,:]
                valid = numpy.where(spec > 0)[0]
                valid = list(range(valid.min(), valid.max()+1))
        else:
            low, high = args.spectral_range.split(',', 1)
            low, high = float(low)*1e6, float(high)*1e6
            valid = numpy.where((freq >= low) & (freq <= high))[0]
            
        # If requested, adjust the frequency range slightly to make prepfold happy
        if args.prepfold:
            which_end = True
            nchan_valid = len(valid)
            while True:
                ## These are the checks that prepfold.c uses
                found = False
                if nchan_valid <= 256:
                    found |= (nchan_valid % 32 == 0)
                    found |= (nchan_valid % 30 == 0)
                    found |= (nchan_valid % 25 == 0)
                    found |= (nchan_valid % 20 == 0)
                elif nchan_valid <= 1024:
                    found |= (nchan_valid % 8 == 0)
                    found |= (nchan_valid % 10 == 0)
                else:
                    found |= (nchan_valid % 128 == 0)
                    found |= (nchan_valid % 100 == 0)
                    
                ## If found, we are done.  Otherwise trim an end and try again.
                if found:
                    break
                    
                if which_end:
                    valid = valid[1:]
                else:
                    valid = valid[:-1]
                which_end ^= True
                nchan_valid = len(valid)
                
        # Sanity checks
        if len(valid) == nchan and args.time_decimation == 1:
           raise RuntimeError("Downselecting does change the file contents, exiting")
        if len(valid) == 0:
            raise RuntimeError("No valid downselected channels found, exiting")
            
        # Report and process
        print("Selecting %i out of a total of %i channels (%.1f%%)" % (len(valid), nchan, 100.*len(valid)/nchan))
        print("->  Frequency range is %.3f to %.3f MHz" % (freq[valid[0]]/1e6, freq[valid[-1]]/1e6))
        
        _fillHDF(hIn, hOut, tDecimation=args.time_decimation, channels=valid)
        
        hIn.close()
        hOut.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='read in DRX/HDF5 waterfall file, pull out the non-zero channels, and decimate the file as requested', 
        epilog='NOTE:  This scripts decimates even if the number of times steps is not an intger multiple of the decimation factor.  This can lead to data loss at the end of observations and at the higher channel numbers.', 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('time_decimation', type=int, 
                        help='temporal decimation factor')
    parser.add_argument('filename', type=str, nargs='+', 
                        help='filename to decimate')
    parser.add_argument('-f', '--force', action='store_true', 
                        help='force overwritting of existing HDF5 files')
    parser.add_argument('-s', '--spectral-range', type=str,
                        help='spectral range to keep in MHz as \'low,high\'')
    parser.add_argument('-p', '--prepfold', action='store_true',
                        help='adjust spectral range to make prepfold happy')
    args = parser.parse_args()
    main(args)
