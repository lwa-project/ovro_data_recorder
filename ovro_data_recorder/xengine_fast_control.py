import copy
import ipaddress

from lwa352_pipeline_control import Lwa352PipelineControl

from mnc.common import NPIPELINE, ETCD_HOST
from mnc.xengine_beamformer_control import AllowedPipelineFailure
from mnc.xengine_beamformer_control import NSERVER, NSTAND

from lwa_antpos.station import ovro, Antenna


NSTAND_FAST = 48


__all__ = ['NSTAND_FAST', 'FastVisibilityControl', 'FastStation']


class FastVisibilityControl(object):
    def __init__(self, servers=None, nserver=8, npipeline_per_server=4, station=ovro, etcdhost=ETCD_HOST):
        # Validate
        assert(nserver <= NSERVER)
        assert(nserver*npipeline_per_server <= NPIPELINE)
        if servers is not None:
            assert(len(servers) <= NSERVER)
            
        # Save the station so that we know what antennas are what
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
                p = Lwa352PipelineControl(hostname, i, etcdhost=etcdhost)
                self.pipelines.append(p)
                
    def set_fast_dest(self, addr_base='10.41.0.76', port_base=11001):
        """
        Set the destination IP address and UDP port for the fast visibility
        data.  Defaults to what is currently used by the "dr-vfast-N" services on Cal/Im.
        """
        
        # Set the address/port pairs for each pipeline
        for i,p in enumerate(self.pipelines):
            ## Two pipelines per output subband
            j = i // 2
            
            addr = ipaddress.IPv4Address(addr_base) + j // 2
            port = port_base + j % 2
            with AllowedPipelineFailure(p):
                p.corr_output_part.set_destination(addr, port)
                
    def set_fast_antennas(self, antennas):
        """
        Given a list of lwa_antpos.station.Antenna instances or digital input
        numbers, set the fast visibility baseline selection to output all four
        polarization pairs for those antennas.
        """
        
        # Validate
        assert(len(antennas) == NSTAND_FAST)
        
        # Antenna to index
        if isinstance(antennas[0], Antenna):
            antennas = [self.station.index(ant) for ant in antennas]
            
        baselines = []
        for i,ant1 in enumerate(antennas):
            for ant2 in enumerate(antennas[i:]):
                baselines.append([[ant1,0],[ant2,0]])
                baselines.append([[ant1,0],[ant2,1]])
                baselines.append([[ant1,1],[ant2,0]])
                baselines.append([[ant1,1],[ant2,1]])
                
        for p in self.pipelines:
            with AllowedPipelineFailure(p):
                p.corr_subsel.set_baseline_select(baselines)
                
    def get_fast_antennas(self, as_index=False):
        """
        Return a list of lwa_antpos.station.Antenna instances for the current
        set of fast visibility baselines.  If `as_index` is True return the list
        as digital input number instead.
        """
        
        # Get the list
        baselines = self.pipelines[0].corr_subsel.get_baseline_select()
        
        # Find the unique antennas
        antennas = []
        for pair in baselines:
            ant1, ant2 = pair
            std1, pol1 = ant1
            std2, pol2 = ant2
            if std1 not in antennas:
                antennas.append(std1)
            if std2 not in antennas: 
                antennas.append(std2)
                
        # Convert to Antenna instances as needed
        if not as_index:
            antennas = [self.station.antennas[ant] for ant in antennas]
            
        return antennas


class FastStation(object):
    """
    Class that wraps FastVisibilityControl to provide a lwa_antpos.station.Station-
    like object for getting the antennas in use.
    """
    
    def __init__(self, servers=None, nserver=8, npipeline_per_server=4, station=ovro, etcdhost=ETCD_HOST):
        self._station = station
        self._control = FastVisibilityControl(servers=servers, nserver=nserver,
                                              npipeline_per_server=npipeline_per_server,
                                              station=station, etcdhost=etcdhost)
        
        # Initial dummy subselected station
        self._substation = self._station.select_subset(list(range(1, NSTAND_FAST+1)))
        
    def refresh(self):
        """
        Refresh the antennas associated with the fast visibility data.
        """
        
        antennas = self._control.get_fast_antennas()
        for i,ant in enumerate(antennas):
            self._substation.antennas[i] = copy.deepcopy(ant)
            
    @property
    def name(self):
        """
        Return the name of the station.
        """
        
        return self._substation.name
        
    @property
    def ecef(self):
        """
        Return the Earth centered, Earth fixed location of the array in meters.
        """
        
        return self._substation.ecef
        
    @property
    def topo_rot_matrix(self):
        """
        Return the rotation matrix that takes a difference in an Earth centered,
        Earth fixed location relative to the Station and rotates it into a
        topocentric frame that is south-east-zenith.
        """
        
        return self._substation.topo_rot_matrix
        
    @property
    def casa_position(self):
        """
        Return a four-element tuple of (CASA position reference, CASA position 1,
        CASA position 2, CASA position 3, CASA position 4) that is suitable for
        use with casacore.measures.measures.position.
        """
        
        return self._substation.casa_position
        
    @property
    def antennas(self):
        return self._substation.antennas
