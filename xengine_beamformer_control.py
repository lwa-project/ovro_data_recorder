import os
import glob
import time
import numpy
import warnings
import progressbar

from lwa352_pipeline_control import Lwa352PipelineControl
from casacore import tables

import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import solar_system_ephemeris, get_body
from astropy.constants import c as speedOfLight
speedOfLight = speedOfLight.to('m/ns').value

from common import NPIPELINE, chan_to_freq
from station import ovro


NCHAN_PIPELINE = 96
NPIPELINE_SUBBAND = 2
NPIPELINE_SERVER = 4

NSUBBAND = NPIPELINE // NPIPELINE_SUBBAND
NSERVER = NPIPELINE // NPIPELINE_SERVER

NSTAND = 352
NPOL = 2


class BeamPointingControl(object):
    """
    Class to provide high level control over beam 1 (and only beam 1 right now).
    """
    
    def __init__(self, nserver=8, npipeline_per_server=4, station=ovro):
        # Validate
        assert(nserver <= NSERVER)
        assert(nserver*npipeline_per_server <= NPIPELINE)
        
        # Save the station so that we know where to point from
        self.station = station
        
        # Contect to the pipelines
        self.pipelines = []
        for s in range(nserver):
            hostname = f"lxdlwagpu{s+1:02d}"
            for i in range(npipeline_per_server):
                p = Lwa352PipelineControl(hostname, i, etcdhost='etcdv3service')
                self.pipelines.append(p)
                
        # Query the pipelines to figure out the frequency ranges they are sending
        self.freqs = []
        for p in self.pipelines:
            metadata = p.beamform.get_bifrost_status()
            freq = chan_to_freq(metadata['chan0'] + numpy.arange(metadata['nchan']))
            self.freqs.append(freq)
            
        # Make a variable to track the per-pipeline calibration state
        self._cal_set = [False for p in self.pipelines]
        
        # Make a variable to keep track of a "gain"
        self._gain = 1.0
        
    @property
    def cal_set(self):
        """
        Boolean as to whether or not all pipelines have had a calibration set.
        """
        
        return all(self._cal_set)
        
    def set_beam1_dest(self, addr='10.41.0.25', port=20001):
        """
        Set the destination IP address and UDP port for the beam data.  Defaults
        to what is currently used by the "dr-beam-1" service on lxdlwagpu09.
        """
        
        for p in self.pipelines:
            p.beamform_output.set_destination([addr] + ['0.0.0.0']*15, [port])
            
    def _freq_to_pipeline(self, freq_to_find):
        """
        Given a frequency in Hz, return the pipeline index where that frequency
        can be found in the data.  Raises a ValueError if there does not appear
        to be a corresponding pipeline.
        """
        
        for i,freq in enumerate(self.freqs):
            if freq_to_find >= freq[0] and freq_to_find <= freq[-1]:
                return i
        raise ValueError(f"Cannot associate {freq_to_find/1e6:.3f} MHz with any pipeline currently under control")
        
    def set_beam1_calibration(self, caltable, verbose=True):
        """
        Given a a CASA measurement set containing a bandpass calibration, load
        the bandpass calibration into the appropriate pipeline(s).
        """
        
        # Validate
        assert(os.path.exists(caltable))
        assert(os.path.isdir(caltable))
        assert(os.path.exists(os.path.join(caltable, 'SPECTRAL_WINDOW')))
        assert(os.path.isdir(os.path.join(caltable, 'SPECTRAL_WINDOW')))
        
        # Load in the calibration data
        tab = tables.table(caltable, ack=False)
        caldata = tab.getcol('CPARAM')[...]
        tab.close()
        
        # Load in the frequency information for the calibration
        tab = tables.table(os.path.join(caltable, 'SPECTRAL_WINDOW'), ack=False)
        calfreq = tab.getcol('CHAN_FREQ')[...]
        calfreq = calfreq.ravel()
        tab.close()
        
        if verbose:
            print(f"Loaded {caldata.shape[0]} by {caldata.shape[1]} by {caldata.shape[2]} complex gains covering {calfreq[0]/1e6:.3f} to {calfreq[-1]/1e6:.3f} MHz")
            
        # Validate the calibration data structure
        assert(caldata.shape[0] == NSTAND)
        assert(caldata.shape[1] == NCHAN_PIPELINE*NPIPELINE_SUBBAND)
        assert(caldata.shape[2] == NPOL)
        
        # Find the pipelines that should correspond to the specified subband
        # TODO: Use the freuqency information to figure this out for the user
        subband_pipelines = []
        for i in range(NPIPELINE_SUBBAND):
            ## Get the frequency range for the pipeline in the subband and pull
            ## out the middle
            center_freq = calfreq[i*NCHAN_PIPELINE:(i+1)*NCHAN_PIPELINE]
            center_freq = center_freq[center_freq.size//2]
            
            ## Try to map that frequency to a pipeline.  If it works, save the
            ## pipeline to subband_pipelines.
            try:
                j = self._freq_to_pipeline(center_freq)
                subband_pipelines.append(self.pipelines[j])
                if verbose:
                    print(f"Found pipeline {j} covering {self.freqs[j][0]/1e6:.3f} to {self.freqs[j][-1]/1e6:.3f} MHz")
            except ValueError:
                pass
                
        # Issue a warning if we don't seem to have the right number of pipelines
        # for the subband
        if len(subband_pipelines) != NPIPELINE_SUBBAND:
            warnings.warn(f"Found {len(subband_pipelines)} pipelines associated with these data instead of the expected {NPIPELINE_SUBBAND}")
            
        # Set the coefficients - this is slow
        pb = progressbar.ProgressBar()
        pb.start(max_value=len(subband_pipelines)*NSTAND)
        for i,p in enumerate(subband_pipelines):
            for j in range(NSTAND):
                for pol in range(NPOL):
                    cal = caldata[j,i*NCHAN_PIPELINE:(i+1)*NCHAN_PIPELINE,pol].ravel()
                    p.beamform.update_calibration_gains(pol, NPOL*j+pol, cal)
                pb += 1
            self._cal_set[i] = True
        pb.finish()
        
    def set_beam1_gain(self, gain):
        """
        Set the "gain" for beam 1 - this is a multiplicative scalar used during
        beamforming.
        
        .. note:: A change to the gain can only activated by an update to the
                  pointing direction.
        """
        
        assert(gain >= 0)
        self._gain = float(gain)
        
    def set_beam1_delays(self, delays, pol=0):
        """
        Set the beamformer delays to the specified values in ns for the given
        polarization.
        """
        
        # Validate
        assert(delays.size == NSTAND*NPOL)
        
        # Make up some dummy amplitudes
        amps = numpy.zeros(NSTAND*NPOL, dtype=numpy.float32)
        amps[pol::NPOL] = self._gain
        # TODO: Remove this when we are done with it
        amps[64:] *= 0
        
        # Set the delays and amplitudes
        pb = progressbar.ProgressBar()
        pb.start(max_value=len(self.pipelines))
        for p in self.pipelines:
            p.beamform.update_delays(pol, delays, amps)
            pb += 1
        pb.finish()
        
    def set_beam1_pointing(self, az, alt, degrees=True):
        """
        Given a topocentric pointing in azimuth and altitude (elevation), point
        the beam in that direction.  The `degrees` keyword determines if the
        coordinates are interpreted in degrees (True) or radians (False).
        """
        
        # Issue a warning if it doesn't look like we've been calibrated
        if not self.cal_set:
            warnings.warn("Calibration is not set, your results may be suspect")
            
        # Convertion from degrees -> radians, plus a validation
        if degrees:
            az *= numpy.pi/180.0
            alt *= numpy.pi/180.0
        assert(az >= 0 and az < 2*numpy.pi)
        assert(alt >= 0 and alt <= numpy.pi/2)
        
        # Figure out what the delays to zenith are
        # TODO: Is this needed?
        zen = numpy.array([0, 0, 1])
        zen_delay = [numpy.dot(zen, ant.enz)/speedOfLight for ant in self.station.antennas]
        
        # Figure out what the delays to the pointing direction are
        dir = numpy.array([numpy.cos(alt)*numpy.sin(az), 
                           numpy.cos(alt)*numpy.cos(az), 
                           numpy.sin(alt)])
        dir_delay = [numpy.dot(dir, ant.enz)/speedOfLight for ant in self.station.antennas]
        
        # Subtract what we need from what we have from the calibration
        # TODO: Is this correct?
        delays = numpy.array(dir_delay) - numpy.array(zen_delay)
        delays = numpy.repeat(delays, NPOL)
        delays = delays.max() - delays
        
        # Apply
        for pol in range(NPOL):
            self.set_beam1_delays(delays, pol)
            
    def set_beam1_target(self, target, verbose=True):
        """
        Given the name of an astronomical target, 'sun', or 'zenith', compute the
        current topocentric position of the body and point the beam at it.
        """
        
        # Figure out what to do with the name
        if target.lower() in ('z', 'zen', 'zenith'):
            ## Zenith is easy
            az, alt = 0.0, 90.0
        else:
            ## Load in where we are
            obs = EarthLocation.from_geocentric(*self.station.ecef, unit=u.m)
            
            ## Resolve the name into coordinates
            if target.lower() in solar_system_ephemeris.bodies:
                if target.lower().startswith('earth'):
                    raise ValueError(f"Invalid target: '{target}'")
                    
                sc = get_body(target.lower(), Time.now(), location=obs)
                if verbose:
                    print(f"Resolved '{target}' to {target.lower()}")
            else:
                sc = SkyCoord.from_name(target)
                if verbose:
                    print(f"Resolved '{target}' to RA {sc.ra}, Dec. {sc.dec}")
                    
            ## Figure out where it is right now
            aa = sc.transform_to(AltAz(obstime=Time.now(), location=obs))
            az = aa.az.deg
            alt = aa.alt.deg
            if verbose:
                print(f"Currently at azimuth {aa.az}, altitude {aa.alt}")
                
        # Point the beam
        self.set_beam1_pointing(az, alt, degrees=True)


class BeamTracker(object):
    """
    Simple class to track a target using a fully calibrated BeamPointingControl
    instance.
    """
    
    def __init__(self, control_instance, update_interval=30):
        self.control_instance = control_instance
        self.update_interval = update_interval
        
    def track(self, target, duration=0, verbose=True):
        """
        Given a target name and a tracking duration in seconds, start tracking
        the source.  If the duration is less than or equal to zero the default
        of 12 hours is used.
        
        .. note:: The tracking can be canceled at any time with a control-C.
        """
        
        # Load in where we are
        obs = EarthLocation.from_geocentric(*self.control_instance.station.ecef, unit=u.m)
        
        # Resolve the name into coordinates
        if target.lower() in solar_system_ephemeris.bodies:
            if target.lower().startswith('earth'):
                raise ValueError(f"Invalid target: '{target}'")
                
            sc = get_body(target.lower(), Time.now(), location=obs)
            if verbose:
                print(f"Resolved '{target}' to {target.lower()}")
        else:
            sc = SkyCoord.from_name(target)
            if verbose:
                print(f"Resolved '{target}' to RA {sc.ra}, Dec. {sc.dec}")
                
        # Figure out the duration of the tracking
        if duration <= 0:
            duration = 86400.0/2
            
        # Set the pointing update time offset - this is half the update interval
        # so that we should end up with a nice scalloped pattern
        puto = TimeDelta(self.update_interval/2.0, format='sec')
        
        # Set the tracking stop time    
        t_stop = time.time() + duration
        
        try:
            # Go!
            t = time.time()
            while t < t_stop:
                ## Get a time marker so that we can account for how long the
                ## update takes
                t_mark = time.time()
                
                ## Figure out where the target will be in puto seconds
                aa = sc.transform_to(AltAz(obstime=Time.now()+puto, location=obs))
                az = aa.az.deg
                alt = aa.alt.deg
                if verbose:
                    print(f"At {time.time():.1f}, moving to azimuth {aa.az}, altitude {aa.alt}")
                    
                ## Point
                self.control_instance.set_beam1_pointing(az, alt, degrees=True)
                
                ## Find how much time we used and when we should sleep until
                t_used = time.time() - t_mark
                t_sleep_until = t + self.update_interval - t_used
                
                ## Sleep to wait it out in increments of 0.5 s so that a control-C
                ## doesn't take forever to register
                while t < t_sleep_until:
                    time.sleep(0.5)
                    t = time.time()
                    
        except KeyboardInterrupt:
            # We gave up, end it now
            pass


def create_and_calibrate(nserver=8, npipeline_per_server=4, cal_directory='/home/ubuntu/mmanders'):
    """
    Wraper to create a new BeamPointingControl instance and load bandpass
    calibration data from a directory.
    """
    
    # Create the instance
    control_instance = BeamPointingControl(nserver=nserver,
                                           npipeline_per_server=npipeline_per_server,
                                           station=ovro)
    
    # Find the calibration files
    calfiles = glob.glob(os.path.join(cal_directory, '*.bcal'))
    calfiles.sort()
    if len(calfiles) == 0:
        warnings.warn(f"No calibration data found in '{cal_directory}'")
        
    # Load the calibration data, if found
    for calfile in calfiles:
        control_instance.set_beam1_calibration(calfile)
       
    # Start up the data flow
    control_instance.set_beam1_dest()
    
    # Done
    return control_instance
