import os
import time
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

from station import ovro

NSERVER = 8
NBAND = 16
NSTAND = 352
NPOL = 2

NCHAN_PIPELINE = 96
NPIPELINE_BAND = 2


class BeamPointingControl(object):
    """
    Class to provide high level control over beam 1 (and only beam 1 right now).
    """
    
    def __init__(self, nserver=8, npipeline_per_server=4, station=ovro):
        # Validate
        assert(nserver <= NSERVER)
        assert(nserver*npipeline_per_server <= NBAND*NPIPELINE_BAND)
        
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
            # TODO: Not so hardcoded
            freq = (metadata['chan0'] + numpy.range(metadata['nchan']))*196e6/8192
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
            p.beamform_output.set_destination([addr] + ['0.0.0.0']*15, [port]) # 1 power beam
            
    def set_beam1_cal(self, band, caltable, verbose=True):
        """
        Given a band number (0 through NBAND-1, inclusive) and a CASA measurement
        set containing a bandpass calibration, load the bandpass calibration into
        the appropriate pipelines.
        """
        
        # Validate
        assert(band >= 0 and band < NBAND)
        assert(os.path.exists(caltable))
        assert(os.path.is_dir(caltable))
        
        # Load in the calibration data
        tab = tables.table(caltab, ack=False)
        caldata = tab.getcol('CPARAM')[...]
        tab.close()
        
        # Load in the frequency information for the calibration
        tab = tables.table(os.path.join(caltab, 'SPECTRAL_WINDOW'))
        freq = tab.getcol('CHAN_FREQ')[...]
        freq = freq.ravel()
        tab.close()
        
        if verbose:
            print(f"Loaded {caldata.shape[0]} by {caldata.shape[1]} by {caldata.shape[2]} complex gains covering {freq[0]/1e6:.3f} to {freq[-1]/1e6:.3f} MHz")
            
        # Validate the calibration data structure
        assert(caldata.shape[0] == NSTAND)
        assert(caldata.shape[1] == NCHAN_PIPELINE*NPIPELINE_BAND)
        assert(caldata.shape[2] == NPOL)
        
        # Find the pipelines that should correspond to the specified band
        # TODO: Use the freuqency information to figure this out for the user
        band_pipelines = []
        for i,p in enumerate(self.pipelines):
            if i//NPIPELINE_BAND == band:
                band_pipelines.append(p)
                
                if verbose:
                    print(f"Found pipeline {i} covering {self.freq[i][0]/1e6:.3f} to {self.freq[i][-1]/1e6:.3f} MHz")
                    
        # Validate that we have the right number of pipelines for the band
        assert(len(band_pipelines) == NPIPELINE_BAND)
        
        # Set the coefficients - this is slow
        pb = progressbar.ProgressBar()
        pb.start(max_value=len(band_pipelines)*NSTAND)
        for i,p in enumerate(band_pipelines):
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
        amps = numpy.zeros(NSTAND*NPOL, dtype=numpy.complex64)
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
