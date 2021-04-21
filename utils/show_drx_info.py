#!/usr/bin/env python3

import os
import sys
import numpy
from datetime import datetime

from lsl.reader import drx, errors


def main(args):
    for filename in args:
        with open(filename, 'rb') as fh:
            frame0 = drx.read_frame(fh)
            fh.seek((os.path.getsize(filename)//drx.FRAME_SIZE-1)*drx.FRAME_SIZE, 0)
            frameN = drx.read_frame(fh)
            fh.seek(0)
            
            print('Time Range:')
            print('  Start:', frame0.time.dp_timetag, '->', frame0.time.datetime)
            print('  Stop: ', frameN.time.dp_timetag, '->', frameN.time.datetime)
            
            sets = []
            times = {}
            tuning_word1, tuning_word2 = None, None
            central_freq1, central_freq2 = None, None
            sample_rate1, sample_rate2 = None, None
            for i in range(256):
                try:
                    frame = drx.read_frame(fh)
                except errors.EOFError:
                    break
                    
                id = frame.id
                if id not in sets:
                    sets.append(id)
                    
                tt = frame.time.dp_timetag
                try:
                    times[id].append(tt)
                except KeyError:
                    times[id] = [tt,]
                    
                beam, tune, pol = id
                if tune == 1:
                    tuning_word1 = frame.payload.tuning_word
                    central_freq1 = frame.central_freq
                    sample_rate1 = frame.sample_rate
                else:
                    tuning_word2 = frame.payload.tuning_word
                    central_freq2 = frame.central_freq
                    sample_rate2 = frame.sample_rate
            sets.sort()
            expected_step = int(round(4096 * (196e6 / sample_rate1)))
            
            print('Frequency Range:')
            if central_freq1 is not None:
                print('  Tuning 1:', tuning_word1, '->', '%.3f MHz' % (central_freq1/1e6,), '@', '%.3f MHz' % (sample_rate1/1e6,))
            else:
                print('  Tuning 1:', 'not found')
            if central_freq2 is not None:
                print('  Tuning 2:', tuning_word2, '->', '%.3f MHz' % (central_freq2/1e6,), '@', '%.3f MHz' % (sample_rate2/1e6,))
            else:
                print('  Tuning 2:', 'not found')
                
            print('Frame Sets:')
            for id in sets:
                print('  Beam %i, tuning %i, pol. %i' % id)
                
            print('Time Flow:')
            for id in sets:
                tts = times[id]
                offsets = {}
                for i in range(1, len(tts)):
                    step = tts[i] - tts[i-1]
                    try:
                        offsets[step] += 1
                    except KeyError:
                        offsets[step] = 1
                print('  Beam %i, tuning %i, pol. %i' % id)
                for step in sorted(list(offsets.keys())):
                    print('    Timetag step of %i:' % step, '%i occurances' % offsets[step], '*' if step == expected_step else '')
                


if __name__ == '__main__':
    main(sys.argv[1:])
