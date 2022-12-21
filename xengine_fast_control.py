import ipaddress

from lwa352_pipeline_control import Lwa352PipelineControl

from mnc.common import NPIPELINE
from mnc.xengine_beamformer_control import AllowedPipelineFailure
from mnc.xengine_beamformer_control import NSERVER, NSTAND

from lwa_antpos.station import ovro, Antenna


class FastVisibilityControl(object):
    def __init__(self, servers=None, nserver=8, npipeline_per_server=4, station=ovro, etcdhost=ETCD_HOST):
        # Validate
        assert(beam in list(range(1,16+1)))
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
        
        # Create the list of IP address/port pairs to send data to
        addrs = []
        ports = []
        for i,p in enumerate(self.pipelines):
            addr.append(ipaddress.IPv4Address(addr_base) + i // 2)
            port.append(port_base + i % 2)
            
        # Set the address/port pairs
        for p,addr,port in zip(self.pipelines, addrs, ports):
            with AllowedPipelineFailure(p):
                p.corr_output_part_control.set_destination(addr, port)
                
    def set_antennas(self, antennas):
        """
        Given a list of lwa_antpos.station.Antenna instances or digital input
        numbers, set the fast visibility baseline selection to output all four
        polarization pairs for those antennas.
        """
        
        # Validate
        assert(len(antennas) == 48)
        
        # Antenna to index
        if isinstance(antennas[0], Antenna):
            antennas = [self.station.index(ant) for ant in antennas]
            
        baselines = []
        for i,ant1 in enumerate(antennas):
            for ant2 in enumerate(antennas[i:]
                baselines.append([[ant1,0],[ant2,0]])
                baselines.append([[ant1,0],[ant2,1]])
                baselines.append([[ant1,1],[ant2,0]])
                baselines.append([[ant1,1],[ant2,1]])
                
        for p in self.pipelines:
            with AllowedPipelineFailure(p):
                p.corr_subsel_control.set_baseline_select(baselines)
                
    def get_antennas(self, as_index=False):
        """
        Return a list of lwa_antpos.station.Antenna instances for the current
        set of fast visibility baselines.  If `as_index` is True return the list
        as digital input number instead.
        """
        
        # Get the list
        baselines = self.pipelines[0].corr_subsel_control.get_baseline_select()
        
        # Find the unique antennas
        antennas = []
        for pair in baseline:
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
