#!/usr/bin/env python3

import os
import sys
import time
import argparse

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-7s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

from astropy.time import TimeDelta

from mnc.common import LWATime, NCHAN as NCHAN_NATIVE, CLOCK as CLOCK_NATIVE
try:
    from mnc.ezdr import Lwa352RecorderControl
except ImportError as e:
    Lwa352RecorderControl = None
    logger.warn(f"Cannot import Lwa352RecorderControl, will not send DR commands")
try:
    from xengine_beamformer_control import BeamPointingControl, BeamTracker
except ImportError as e:
    BeamPointingControl = BeamTracker = None
    logger.warn(f"Cannot import BeamPointingControl, will not send beamformer commands")


def _tag_to_step(tag):
    """
    Convert a OBS_STP_* tag into a zero-based step index.
    """
    
    step = tag.split('[', 1)[1]
    step = int(step.split(']', 1)[0], 10) - 1
    return step


def parse_sdf(filename):
    """
    Given an SDF beamformer filename, parse the file and return a list of
    dictionaries that describe all of pointing and dwell times that we needed.
    
    The dictionaries contain:
     * 'mjd' - the start time of the observation as an integer MJD
     * 'mpm' - the start time of the observation as an integer ms past midnight
     * 'dur' - the observation duration in ms
     * 'ra'  - the target RA in hours, J2000 -or- the Sun or Jupiter
     * 'dec' - the target Dec in degrees, J2000 -or- None for the Sun/Jupiter
    """
    
    # Parse the SDF in a simple way that only keeps track of the pointing/durations
    obs = []
    time_avg = 0
    tint_native = 2*NCHAN_NATIVE / CLOCK_NATIVE * 24
    with open(filename, 'r') as fh:
        for line in fh:
            line = line.strip().rstrip()
            
            if line.startswith('OBS_ID'):
                ## New observation?
                try:
                    obs.append(temp)
                except NameError:
                    pass
                temp = {}
            elif line.startswith('OBS_START_MJD'):
                ## Observation start MJD
                temp['mjd'] = int(line.rsplit(None, 1)[1], 10)
            elif line.startswith('OBS_START_MPM'):
                ## Observation start MPM
                temp['mpm'] = int(line.rsplit(None, 1)[1], 10)
            elif line.startswith('OBS_DUR') and not line.startswith('OBS_DUR+'):
                ## Observation duration in ms
                temp['dur'] = int(line.rsplit(None, 1)[1], 10)
            elif line.startswith('OBS_MODE'):
                ## Observation mode - this can fail if we are asked to do something
                ## we cannot.
                mode = line.rsplit(None, 1)[1]
                if mode == 'TRK_SOL':
                    ### Looking at the Sun is ok
                    temp['ra'] = 'sun'
                    temp['dec'] = None
                elif mode == 'TRK_JOV':
                    ### Looking at Jupiter is ok
                    temp['ra'] = 'jupiter'
                    temp['dec'] = None
                elif mode in ('TRK_RADEC', 'STEPPED'):
                    ### Tracking a point on the sky is ok as is a stepped observation
                    ### so long as it is in RA/dec. (more on that later).
                    pass
                else:
                    raise RuntimeError(f"Invalid observing mode '{mode}'")
            elif line.startswith('OBS_RA'):
                ## RA to track in hours, J2000
                temp['ra'] = float(line.rsplit(None, 1)[1])
            elif line.startswith('OBS_DEC'):
                ## Dec. to track in degrees, J2000
                temp['dec'] = float(line.rsplit(None, 1)[1])
            elif line.startswith('OBS_STP_RADEC'):
                ## Stepped mode test - we only support RA/dec at this time
                mode = int(line.rsplit(None, 1)[1], 10)
                if mode != 1:
                    raise RuntimeError(f"Invalid observing mode '{line}'")
            elif line.startswith('OBS_STP_N'):
                ## Stepped mode setup
                count = int(line.rsplit(None, 1)[1], 10)
                temp['steps'] = [{'ra':None, 'dec': None, 'dur':0} for i in range(count)]
            elif line.startswith('OBS_STP_C1'):
                ## Stepped mode ra in hours, J2000
                tag, ra = line.rsplit(None, 1)
                step = _tag_to_step(tag)
                ra = float(ra)
                for i in range(step, len(temp['steps'])):
                    temp['steps'][i]['ra'] = ra
            elif line.startswith('OBS_STP_C2'):
                ## Stepped mode dec. in degrees, J2000
                tag, dec = line.rsplit(None, 1)
                step = _tag_to_step(tag)
                dec = float(dec)
                for i in range(step, len(temp['steps'])):
                    temp['steps'][i]['dec'] = dec
            elif line.startswith('OBS_STP_T'):
                ## Stepped mode duration in ms
                tag, dur = line.rsplit(None, 1)
                step = _tag_to_step(tag)
                dur = int(dur, 10)
                for i in range(step, len(temp['steps'])):
                    temp['steps'][i]['dur'] = dur
            elif line.startswith('SESSION_DRX_BEAM'):
                beam = int(line.rsplit(None, 1)[1], 10)
                if beam != 1:
                    logger.warn(f"Beam {beam} requested but observation will run on beam 1")
            elif line.startswith('SESSION_SPC'):
                _, nchan, nwin = line.rsplit(None, 2)
                nchan = int(nchan, 10)
                nwin = int(nwin, 10)
                tint = (nchan*nwin / 19.6e6)
                time_avg = int(round(tint / tint_native))
                while 1000 % time_avg != 0 and time_avg < 1000:
                    time_avg += 1
                time_avg = min([time_avg, 1000])
                
        # Add on that last observation
        obs.append(temp)
        
    # Expand the steps
    expanded_obs = []
    for o in obs:
        try:
            ## First step
            expanded_obs.append({'mjd': o['mjd'],
                                 'mpm': o['mpm'],
                                 'dur': o['steps'][0]['dur'],
                                 'ra':  o['steps'][0]['ra'],
                                 'dec': o['steps'][0]['dec']})
            ## Steps 2 through N so that the start time builds off the previous
            ## step's duration
            for s in o['steps'][1:]:
                expanded_obs.append({'mjd': expanded_obs[-1]['mjd'],
                                     'mpm': expanded_obs[-1]['mpm']+expanded_obs[-1]['dur'],
                                     'dur': s['dur'],
                                     'ra':  s['ra'],
                                     'dec': s['dec']})
            
        except KeyError:
            ## Nope, it's a normal observation
            expanded_obs.append(o)
            
    # One last check and save the averaging time
    if time_avg > 0:
        if round(tint, 3) != round(time_avg*tint_native, 3):
            logger.warn(f"Requested {tint*1000:.1f} ms spectrometer time resolution will not but used, {time_avg*tint_native*1000:.1f} ms will be used instead")
    else:
        time_avg = 1
        logger.warn(f"Spectrometer mode was not requested but will be used anyway with an integration time of {tint_native*1000:.1f} ms")
    expanded_obs[0]['time_avg'] = time_avg
    
    return expanded_obs


def main(args):
    # Parse the SDF
    obs = parse_sdf(args.filename)
    logger.info(f"Loaded '{os.path.basename(args.filename)}' with a sequence of {len(obs)} pointing(s)")
    
    # Figure out the start and stop times for the actual run as well as when we
    # should start and stop the recording
    ## Observation
    start = LWATime(obs[0]['mjd'], obs[0]['mpm']/1000/86400, format='mjd', scale='utc')
    stop  = LWATime(obs[-1]['mjd'], (obs[-1]['mpm'] + obs[-1]['dur'])/1000/86400, format='mjd', scale='utc')
    dur = (stop - start).sec
    logger.info(f"Observations start at {start.datetime} and continue for {dur:.3f} s")
    
    ## Recording - start 5 s before and goes for 5 s afterwards
    rec_start = start - TimeDelta(5, format='sec')
    rec_stop = stop + TimeDelta(5, format='sec')
    rec_dur = (rec_stop - rec_start).sec
    logger.info(f"Recording starts at {rec_start.datetime} and continutes for {rec_dur:.3f} s")
    
    ## Validate
    if rec_start > LWATime.now() - TimeDelta(30, format='sec'):
        logger.error("Insufficient advanced notice to run this SDF, aborting")
        sys.exit(1)
    elif rec_start < LWATime.now():
        logger.error("SDF appears to be in the past, aborting")
        sys.exit(1)
        
    # Setup the control
    ## Recording
    try:
        dr = Lwa352RecorderControl('power')
    except Exception:
        logger.warn(f"Cannot create DR control object, will not send DR commands")
        dr = None
    ## Beamforming
    try:
        bf = BeamPointingControl()
        bt = BeamTracker(bf, update_interval=15)
    except Exception:
        logger.warn(f"Cannot create beamformer control objects, will not send beamformer commands")
        bf = None
        bt = None
        
    # Recording
    ## Wait for the right time
    logger.info("Waiting for the recording time...")
    while LWATime.now() < rec_start - TimeDelta(15, format='sec'):
        time.sleep(0.01)
        
    ## Schedule it
    logger.info("Sending recorder command")
    if dr is not None and not args.dry_run:
        dr.record(start_mjd=obs[0]['mjd'], start_mpm=obs[0]['mpm']-5000, dur=rec_dur, time_avg=obs[0]['time_avg'])
        
    # Beamforming/tracking
    ## Wait for the right time
    logger.info("Waiting for the start of the first observation...")
    ## TODO: Wait time correction for the pipeline lag?
    while LWATime.now() < start:
        time.sleep(0.01)
        
    ## Iterate through the observations
    for i,o in enumerate(obs):
        name = o['ra']
        if o['dec'] is not None:
            name = f"{o['ra']} hr, {o['dec']} deg"
        logger.info(f"Tracking pointing #{i+1} ('{name}') for {o['dur']/1000.0:.3f} s")
        if bt is not None and not args.dry_run:
            bt.track(o['ra'], dec=o['dec'], duration=o['dur']/1000.0)
            
    # Close it out
    logger.info("Finished with the observations, waiting for the recording to finish...")
    while LWATime.now() < rec_stop:
        time.sleep(0.01)
    logger.info("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='read in a SDF file and convert it to a sequence of commands to run',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('filename', type=str,
                        help='filename to parse and run')
    parser.add_argument('-n', '--dry-run', action='store_true',
                        help='parse and print commands but do not send them')
    args = parser.parse_args()
    main(args)
