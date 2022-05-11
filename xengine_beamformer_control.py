import os
import glob
import json
import time
import numpy
import warnings
import progressbar
from threading import RLock
from textwrap import fill as tw_fill
from concurrent.futures import ThreadPoolExecutor, wait as thread_pool_wait

from lwa352_pipeline_control import Lwa352PipelineControl
from casacore import tables

import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, Angle, EarthLocation, AltAz
from astropy.coordinates import solar_system_ephemeris, get_body
from astropy.constants import c as speedOfLight
speedOfLight = speedOfLight.to('m/ns').value

from lwa_antpos.station import ovro

from mnc.common import NPIPELINE, chan_to_freq


NCHAN_PIPELINE = 96
NPIPELINE_SUBBAND = 2
NPIPELINE_SERVER = 4

NSUBBAND = NPIPELINE // NPIPELINE_SUBBAND
NSERVER = NPIPELINE // NPIPELINE_SERVER

NSTAND = 352
NPOL = 2


def _build_repr(name, attrs=[]):
    name = '.'.join(name.split('.')[-2:])
    output = "<%s" % name
    first = True
    for key,value in attrs:
        output += "%s %s=%s" % (('' if first else ','), key, value)
        first = False
    output += ">"
    return output


class AllowedPipelineFailure(object):
    """
    Context manager to ignore failures while controlling the pipelines.
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            warnings.warn(f"Failed to command {self.pipeline.host} (pipeline {self.pipeline.pipeline_id})")
        return True


class PipelineTaskPool(list):
    def __init__(self, objects=[]):
        list.__init__(self, objects)
        
    def __getattr__(self, item):
        with ThreadPoolExecutor(8) as pool:
            results = {}
            for i,obj in enumerate(self):
                results[pool.submit(lambda x: getattr(x, item), obj)] = i
        output = PipelineTaskPool([None,]*len(self))
        for result in thread_pool_wait(results).done:
            i = results[result]
            output[i] = result.result()
        return output
        
    def __call__(self, *args, **kwds):
        with ThreadPoolExecutor(8) as pool:
            results = {}
            for i,obj in enumerate(self):
                results[pool.submit(lambda x: x.__call__(*args, **kwds), obj)] = i
        output = [None,]*len(self)
        for result in thread_pool_wait(results).done:
            i = results[result]
            with AllowedPipelineFailure(self[i].__self__):
                output[i] = result.result()
        return output


_BEAM_DEST_LOCK = RLock()


class BeamPointingControl(object):
    """
    Class to provide high level control over a beam.
    """
    
    def __init__(self, beam, servers=None, nserver=8, npipeline_per_server=4, station=ovro):
        # Validate
        assert(beam in list(range(1,16+1)))
        assert(nserver <= NSERVER)
        assert(nserver*npipeline_per_server <= NPIPELINE)
        if servers is not None:
            assert(len(servers) <= NSERVER)
            
        # Save the beam
        self.beam = beam
        
        # Save the station so that we know where to point from
        self.station = station
        
        # Figure out the servers to control
        if servers is None:
            servers = []
            for s in range(nserver):
                servers.append(f"lxdlwagpu{s+1:02d}")
                
        # Contect to the pipelines
        self.pipelines = []
        for hostname in servers:
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
        
        # Initially set uniform antenna weighting for a natural beam shape
        self.set_beam_weighting(lambda x: 1.0)
        
    def __repr__(self):
        n = self.__class__.__module__+'.'+self.__class__.__name__
        a = [(attr,getattr(self, attr, None)) for attr in ('beam',)]
        return tw_fill(_build_repr(n,a), subsequent_indent='    ')
        
    @property
    def cal_set(self):
        """
        Boolean as to whether or not all pipelines have had a calibration set.
        """
        
        return all(self._cal_set)
        
    def set_beam_dest(self, addr='10.41.0.25', port=None, port_base=20001):
        """
        Set the destination IP address and UDP port for the beam data.  Defaults
        to what is currently used by the "dr-beam-N" services on lxdlwagpu09.
        """
        
        # If a port was not explicitly provided, find what is should be using
        # port_base and the beam number.
        if port is None:
            port = port_base + self.beam - 1
            
        with _BEAM_DEST_LOCK:
            for p in self.pipelines:
                with AllowedPipelineFailure(p):
                    # Get the current settings for the pipeline
                    metadata = p.beamform_output.get_bifrost_status()
                    addrs = json.loads(metadata['stats']['dest_ip'].replace("'", '"'))
                    ports = json.loads(metadata['stats']['dest_port'])
                    while len(addrs) < 16:
                        addrs.extend(addrs)
                    addrs = addrs[:16]
                    while len(ports) < 16:
                        ports.extend(ports)
                    ports = ports[:16]
                    # Update this beam
                    addrs[self.beam-1] = addr
                    ports[self.beam-1] = port
                    # Send them out again
                    p.beamform_output.set_destination(addrs, ports)
                    
    def set_beam_vlbi_dest(self, addr='10.41.0.73', port=21001):
        """
        Set the destination IP address and UDP port for the VLBI version of the
        beam data.  Defaults to what is currently used by the "dr-tengine"
        service on lxdlwagpu09.
        """
        
        # Validate that this is the correct beam
        if self.beam != 1:
            raise RuntimeError("The VLBI beam is controlled through beam 1")
            
        for p in self.pipelines:
            with AllowedPipelineFailure(p):
                p.beamform_vlbi_output.set_destination(addr, port)
                
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
        
    def set_beam_calibration(self, caltable, verbose=True):
        """
        Given a a CASA measurement set containing a bandpass calibration, load
        the bandpass calibration into the appropriate pipeline(s).
        """
        
        # Validate
        assert(os.path.exists(caltable))
        assert(os.path.isdir(caltable))
        assert(os.path.exists(os.path.join(caltable, 'SPECTRAL_WINDOW')))
        assert(os.path.isdir(os.path.join(caltable, 'SPECTRAL_WINDOW')))
        
        # Load in the calibration data and normalize it
        tab = tables.table(caltable, ack=False)
        caldata = tab.getcol('CPARAM')[...]
        caldata /= numpy.abs(caldata)
        
        # Load in the flagging data for the calibration
        flgdata = tab.getcol('FLAG')[...]
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
        subband_pipeline_index = []
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
                subband_pipeline_index.append(j)
                if verbose:
                    print(f"Found pipeline {j} covering {self.freqs[j][0]/1e6:.3f} to {self.freqs[j][-1]/1e6:.3f} MHz")
            except ValueError:
                pass
                
        # Issue a warning if we don't seem to have the right number of pipelines
        # for the subband
        if len(subband_pipelines) != NPIPELINE_SUBBAND:
            warnings.warn(f"Found {len(subband_pipelines)} pipelines associated with these data instead of the expected {NPIPELINE_SUBBAND}")
            
        # Set the coefficients - this is slow
        pb = progressbar.ProgressBar(redirect_stdout=True)
        pb.start(max_value=len(subband_pipelines)*NSTAND)
        for i,(p,ii) in enumerate(zip(subband_pipelines, subband_pipeline_index)):
            for j in range(NSTAND):
                for pol in range(NPOL):
                    cal = 1./caldata[j,i*NCHAN_PIPELINE:(i+1)*NCHAN_PIPELINE,pol].ravel()
                    cal = numpy.where(numpy.isfinite(cal), cal, 0)
                    flg = flgdata[j,i*NCHAN_PIPELINE:(i+1)*NCHAN_PIPELINE,pol].ravel()
                    cal *= (1-flg)
                    
                    with AllowedPipelineFailure(p):
                        p.beamform.update_calibration_gains(2*(self.beam-1)+pol, NPOL*j+pol, cal)
                        time.sleep(0.005)
                pb += 1
            self._cal_set[ii] = True
        pb.finish()
        
    def set_beam_gain(self, gain):
        """
        Set the "gain" for beam 1 - this is a multiplicative scalar used during
        beamforming.
        
        .. note:: A change to the gain can only activated by an update to the
                  pointing direction.
        """
        
        assert(gain >= 0)
        self._gain = float(gain)
        
    def set_beam_weighting(self, fnc=lambda x: 1.0):
        """
        Set the beamformer antenna weighting using the provided function.  The
        function should accept a single floating point input of an antenna's
        distance from the array center (in meters) and return a weight between
        0 and 1, inclusive.
        """
        
        fnc2 = lambda x: numpy.clip(fnc(numpy.sqrt(x[0]**2 + x[1]**2)), 0, 1)
        self._weighting = numpy.array([fnc2(ant.enz) for ant in self.station.antennas])
        self._weighting = numpy.repeat(self._weighting, 2)
        
    def set_beam_delays(self, delays, pol=0, load_time=None):
        """
        Set the beamformer delays to the specified values in ns for the given
        polarization.
        """
        
        # Validate
        assert(delays.size == NSTAND*NPOL)
        
        # Make up some dummy amplitudes
        amps = numpy.zeros(NSTAND*NPOL, dtype=numpy.float32)
        amps[pol::NPOL] = self._gain
        amps *= self._weighting
        
        # Set the delays and amplitudes
        ptp = PipelineTaskPool(self.pipelines)
        ptp.beamform.update_delays(2*(self.beam-1)+pol, delays, amps, load_time=load_time, time_unit='time')
        
    def set_beam_pointing(self, az, alt, degrees=True, load_time=None):
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
            self.set_beam_delays(delays, pol, load_time=load_time)
            
    def set_beam_target(self, target_or_ra, dec=None, load_time=None, verbose=True):
        """
        Given the name of an astronomical target, 'sun', or 'zenith', compute the
        current topocentric position of the body and point the beam at it.  If
        the 'dec' keyword is not None, the target is intepreted to be a RA.
        """
        
        # Force to string
        target_or_ra = str(target_or_ra)
        if dec is not None:
            dec = str(dec)
            
        # Figure out what to do with the name
        if target_or_ra.lower() in ('z', 'zen', 'zenith'):
            ## Zenith is easy
            az, alt = 0.0, 90.0
        else:
            ## Load in where we are
            obs = EarthLocation.from_geocentric(*self.station.ecef, unit=u.m)
            
            ## Resolve the name into coordinates
            if dec is not None:
                ra = Angle(target_or_ra, unit='hourangle')
                dec = Angle(dec, unit='deg')
                sc = SkyCoord(ra, dec, frame='fk5')
                if verbose:
                    print(f"Resolved '{target_or_ra}, {dec}' to RA {sc.ra}, Dec. {sc.dec}")
                    
            elif target_or_ra.lower() in solar_system_ephemeris.bodies:
                if target_or_ra.lower().startswith('earth'):
                    raise ValueError(f"Invalid target: '{target_or_ra}'")
                    
                sc = get_body(target_or_ra.lower(), Time.now(), location=obs)
                if verbose:
                    print(f"Resolved '{target_or_ra}' to {target_or_ra.lower()}")
            else:
                sc = SkyCoord.from_name(target_or_ra)
                if verbose:
                    print(f"Resolved '{target_or_ra}' to RA {sc.ra}, Dec. {sc.dec}")
                    
            ## Figure out where it is right now
            aa = sc.transform_to(AltAz(obstime=Time.now(), location=obs))
            az = aa.az.deg
            alt = aa.alt.deg
            if verbose:
                print(f"Currently at azimuth {aa.az}, altitude {aa.alt}")
                
        # Point the beam
        self.set_beam_pointing(az, alt, degrees=True, load_time=load_time)


class BeamTracker(object):
    """
    Simple class to track a target using a fully calibrated BeamPointingControl
    instance.
    """
    
    def __init__(self, control_instance, update_interval=30):
        if not isinstance(control_instance, BeamPointingControl):
            raise ValueError("Expected control_instance to be of type BeamPointingControl")
        self.control_instance = control_instance
        self.update_interval = update_interval
        
    def __repr__(self):
        n = self.__class__.__module__+'.'+self.__class__.__name__
        a = [(attr,getattr(self, attr, None)) for attr in ('control_instance', 'update_interval')]
        return tw_fill(_build_repr(n,a), subsequent_indent='    ')
        
    def track(self, target_or_ra, dec=None, duration=0, start_time=None, verbose=True):
        """
        Given a target name and a tracking duration in seconds, start tracking
        the source.  If the 'dec' keyword is not None, the target is intepreted
        to be a RA.  If the duration is less than or equal to zero the default
        of 12 hours is used.
        
        .. note:: The tracking can be canceled at any time with a control-C.
        """
        
        # Force to string
        target_or_ra = str(target_or_ra)
        if dec is not None:
            dec = str(dec)
            
        # Load in where we are
        obs = EarthLocation.from_geocentric(*self.control_instance.station.ecef, unit=u.m)
        
        # Resolve the name into coordinates
        if target_or_ra.lower() in solar_system_ephemeris.bodies:
            if target_or_ra.lower().startswith('earth'):
                raise ValueError(f"Invalid target: '{target_or_ra}'")
                
            sc = get_body(target_or_ra.lower(), Time.now(), location=obs)
            if verbose:
                print(f"Resolved '{target_or_ra}' to {target_or_ra.lower()}")
        else:
            if dec is not None:
                ra = Angle(target_or_ra, unit='hourangle')
                dec = Angle(dec, unit='deg')
                sc = SkyCoord(ra, dec, frame='fk5')
                if verbose:
                    print(f"Resolved '{target_or_ra}, {dec}' to RA {sc.ra}, Dec. {sc.dec}")
            else:
                sc = SkyCoord.from_name(target_or_ra)
                if verbose:
                    print(f"Resolved '{target_or_ra}' to RA {sc.ra}, Dec. {sc.dec}")
                    
        # Figure out the duration of the tracking
        if duration <= 0:
            duration = 86400.0/2
            
        # Set the pointing update time offset - this is half the update interval
        # so that we should end up with a nice scalloped pattern
        puto = TimeDelta(min([duration, self.update_interval])/2.0, format='sec')
        
        # Set the tracking stop time    
        if start_time is None:
            start_time = time.time()
        t_stop = start_time + duration
        
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
                self.control_instance.set_beam_pointing(az, alt, degrees=True, load_time=t_mark+2)
                
                ## Find how much time we used and when we should sleep until
                t_used = time.time() - t_mark
                t_sleep_until = t + self.update_interval - t_used
                
                ## Sleep to wait it out in increments of 0.01 s so that a control-C
                ## doesn't take forever to register
                while t < t_sleep_until and t < t_stop:
                    time.sleep(0.01)
                    t = time.time()
                    
        except KeyboardInterrupt:
            # We gave up, end it now
            pass


def create_and_calibrate(beam, servers=None, nserver=8, npipeline_per_server=4, cal_directory='/home/ubuntu/mmanders/caltables/latest/'):
    """
    Wraper to create a new BeamPointingControl instance and load bandpass
    calibration data from a directory.
    """
    
    # Create the instance
    control_instance = BeamPointingControl(beam,
                                           servers=servers,
                                           nserver=nserver,
                                           npipeline_per_server=npipeline_per_server,
                                           station=ovro)
    
    # Find the calibration files
    calfiles = glob.glob(os.path.join(cal_directory, '*.bcal'))
    calfiles.sort()
    if len(calfiles) == 0:
        warnings.warn(f"No calibration data found in '{cal_directory}'")
        
    # Load the calibration data, if found
    for calfile in calfiles:
        control_instance.set_beam_calibration(calfile)
       
    # Start up the data flow
    control_instance.set_beam_dest()
    #if control_instance.beam == 1:
    #    control_instance.set_beam_vlbi_dest()
        
    # Done
    return control_instance
