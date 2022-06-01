#!/usr/bin/env python3

import os
import sys
import time
import numpy
import argparse

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-7s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, Angle, EarthLocation, AltAz
from astropy.coordinates import solar_system_ephemeris, get_body

from lwa_antpos.station import ovro

from mnc.mcs import Client as MCSClient
from mnc.common import LWATime, NCHAN as NCHAN_NATIVE, CLOCK as CLOCK_NATIVE
try:
    from xengine_beamformer_control import BeamPointingControl
except ImportError as e:
    BeamPointingControl = BeamTracker = None
    logger.warn(f"Cannot import BeamPointingControl, will not send beamformer commands")


# Beam tracking update control
#: Time step to use when determining beam pointings
_BEAM_UPDATE_STEP = TimeDelta(1, format='sec')
#: Maximum distance a source can move on the sky before a beam pointing update
_BEAM_UPDATE_DISTANCE = Angle('0:04:48', unit='deg') # This set based on LWA1's MCS
                                                     # step size of 0.2 degrees and
                                                     # scaling that to an aperature
                                                     # of 250 m.


def _tag_to_step(tag):
    """
    Convert a OBS_STP_* tag into a zero-based step index.
    """
    
    step = tag.split('[', 1)[1]
    step = int(step.split(']', 1)[0], 10) - 1
    return step


def _separation(aa0, aa1):
    """
    Wrapper around the astropy angular_separation function that takes in two
    AzAlt instances and returns the angular distance between them.
    
    From:
      https://github.com/astropy/astropy/blob/main/astropy/coordinates/angle_utilities.py
    """
    
    lon1, lat1 = aa0.az.rad, aa0.alt.rad
    lon2, lat2 = aa1.az.rad, aa1.alt.rad
    
    sdlon = numpy.sin(lon2 - lon1)
    cdlon = numpy.cos(lon2 - lon1)
    slat1 = numpy.sin(lat1)
    slat2 = numpy.sin(lat2)
    clat1 = numpy.cos(lat1)
    clat2 = numpy.cos(lat2)
    
    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon
    denominator = slat1 * slat2 + clat1 * clat2 * cdlon
    
    return Angle(numpy.arctan2(numpy.hypot(num1, num2), denominator), unit='rad').to('deg')


def _get_tracking_updates(obs):
    """
    Given an observations dictionary, figure out when/how to update the beam
    pointing for the duration of the observation.  Returns a list of times/
    pointings (unix timestamp for the update, azimuth in degrees, elevation in
    degrees).
    """
    
    # Load the site
    site = EarthLocation.from_geocentric(*ovro.ecef, unit=u.m)
    
    # Load the start and stop times for this observation
    start = LWATime(obs['mjd'], obs['mpm']/1000/86400, format='mjd', scale='utc')
    stop =  LWATime(obs['mjd'], (obs['mpm']+obs['dur'])/1000/86400, format='mjd', scale='utc')
    
    # az/alt vs ra/dec step check - If we are in az/alt mode we only need to
    # point once at the beginning of the observation
    if obs['azalt']:
        return [[start.utc.unix, obs['ra'], obs['dec']],]
        
    # Load in the target - coordinates or a solar system body
    target_or_ra = obs['ra']
    dec = obs['dec']
    if dec is not None:
        ra = Angle(target_or_ra, unit='hourangle')
        dec = Angle(dec, unit='deg')
        sc = SkyCoord(ra, dec, frame='fk5')
    else:
        sc = get_body(target_or_ra.lower(), start+(stop-start)/2, location=site)
        
    # Figure out when to update based on how far the sources moves
    t = start
    updates = [t,]
    aa = sc.transform_to(AltAz(obstime=t, location=site))
    last_azalt = aa
    while t <= stop:
        aa = sc.transform_to(AltAz(obstime=t, location=site))
        if _separation(aa, last_azalt) >= _BEAM_UPDATE_DISTANCE:
            updates.append(t)
            last_azalt = aa
        t += _BEAM_UPDATE_STEP
    updates.append(stop)
    
    # Convert to updates into pointings that happen at t[i] but point to the
    # position of the source at the midpoint between t[i] and t[i+1].
    steps = []
    for i in range(len(updates)-1):
        t_diff = updates[i+1] - updates[i]
        t_step = updates[i] + t_diff / 2.0
        aa = sc.transform_to(AltAz(obstime=t_step, location=site))
        steps.append([updates[i].utc.unix, aa.az.deg, aa.alt.deg])
        
    # Done
    return steps


def parse_sdf(filename):
    """
    Given an SDF beamformer filename, parse the file and return a list of
    dictionaries that describe all of pointing and dwell times that we needed.
    
    The dictionaries contain:
     * 'mjd'   - the start time of the observation as an integer MJD
     * 'mpm'   - the start time of the observation as an integer ms past midnight
     * 'dur'   - the observation duration in ms
     * 'ra'    - the target RA in hours, J2000 -or- the Sun or Jupiter
     * 'dec'   - the target Dec in degrees, J2000 -or- None for the Sun/Jupiter
     * 'azalt' - if 'ra' and 'dec' are actually azimuth and elevation, both in
                 degrees
    """
    
    # Parse the SDF in a simple way that only keeps track of the pointing/durations
    obs = []
    beam = 1
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
                temp['azalt'] = False
            elif line.startswith('OBS_RA'):
                ## RA to track in hours, J2000
                temp['ra'] = float(line.rsplit(None, 1)[1])
            elif line.startswith('OBS_DEC'):
                ## Dec. to track in degrees, J2000
                temp['dec'] = float(line.rsplit(None, 1)[1])
            elif line.startswith('OBS_STP_RADEC'):
                ## Stepped mode test - toggle azalt as needed
                mode = int(line.rsplit(None, 1)[1], 10)
                if mode != 1:
                    temp['azalt'] = True
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
            expanded_obs.append({'mjd':   o['mjd'],
                                 'mpm':   o['mpm'],
                                 'dur':   o['steps'][0]['dur'],
                                 'ra':    o['steps'][0]['ra'],
                                 'dec':   o['steps'][0]['dec'],
                                 'azalt': o['azalt']})
            ## Steps 2 through N so that the start time builds off the previous
            ## step's duration
            for s in o['steps'][1:]:
                expanded_obs.append({'mjd':   expanded_obs[-1]['mjd'],
                                     'mpm':   expanded_obs[-1]['mpm']+expanded_obs[-1]['dur'],
                                     'dur':   s['dur'],
                                     'ra':    s['ra'],
                                     'dec':   s['dec'],
                                     'azalt': expanded_obs[-1]['azalt']})
            
        except KeyError:
            ## Nope, it's a normal observation
            expanded_obs.append(o)
            
    # One last check and save the averaging time and beam
    if time_avg > 0:
        if round(tint, 3) != round(time_avg*tint_native, 3):
            logger.warn(f"Requested {tint*1000:.1f} ms spectrometer time resolution will not but used, {time_avg*tint_native*1000:.1f} ms will be used instead")
    else:
        time_avg = 1
        logger.warn(f"Spectrometer mode was not requested but will be used anyway with an integration time of {tint_native*1000:.1f} ms")
    expanded_obs[0]['time_avg'] = time_avg
    expanded_obs[0]['beam'] = beam
    
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
    if rec_start < LWATime.now() + TimeDelta(30, format='sec'):
        logger.error("Insufficient advanced notice to run this SDF, aborting")
        sys.exit(1)
    elif rec_start < LWATime.now():
        logger.error("SDF start time appears to be in the past, aborting")
        sys.exit(1)
        
    # Setup the beamformer stepping
    for o in obs:
        o['sdf_steps'] = _get_tracking_updates(o)
        for step in o['sdf_steps']:
            logger.debug(f"Beam to az {step[1]:.3f} deg, alt {step[2]:.3f} deg at {step[0]:.3f}")
            
    # Setup the control
    ## Recording
    try:
        dr = MCSClient()
    except Exception:
        logger.warn("Cannot create DR control object, will not send DR commands")
        dr = None
    ## Beamforming
    try:
        bf = BeamPointingControl(obs[0]['beam'])
    except Exception:
        logger.warn("Cannot create beamformer control object, will not send beamformer commands")
        bf = None
        
    # Recording
    ## Wait for the right time
    logger.info("Waiting for the recording time...")
    while LWATime.now() < rec_start - TimeDelta(15, format='sec'):
        time.sleep(0.01)
        
    ## Schedule it
    logger.info("Sending recorder command")
    if dr is not None and not args.dry_run:
        status = dr.send_command(f"dr{obs[0]['beam']}", 'record',
                                 start_mjd=obs[0]['mjd'],
                                 start_mpm=obs[0]['mpm'],
                                 duration_ms=int(rec_dur*1000),
                                 time_avg=obs[0]['time_avg'])
        if status[0]:
            logger.info("Record command succeeded: %s" % str(status[1:]))
        else:
            logger.error("Record command failed: %s", str(status[1]))
            
    # Beamforming/tracking
    ## Wait for the right time
    logger.info("Waiting for the start of the first observation...")
    while LWATime.now() + TimeDelta(1, format='sec') < start:
        time.sleep(0.01)
        
    ## Iterate through the observations
    for i,o in enumerate(obs):
        name = o['ra']
        if o['dec'] is not None:
            name = f"{o['ra']} hr, {o['dec']} deg"
        logger.info(f"Tracking pointing #{i+1} ('{name}') for {o['dur']/1000.0:.3f} s")
        for step in o['sdf_steps']:
            while time.time() + 1 < step[0]:
                time.sleep(0.01)
            if bf is not None and not args.dry_run:
                bf.set_beam_pointing(step[1], step[2], degrees=True, load_time=step[0])
                
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
