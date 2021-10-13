import os
import sys
import copy
import numpy
import weakref

from astropy.coordinates import EarthLocation
import astropy.units as u

__all__ = ['OVRO_CONFIG_FILENAME', 'Station', 'Antenna', 'parse_config', 'ovro']


OVRO_CONFIG_FILENAME = os.path.join(os.path.dirname(__file__), 'ovro.txt')


def _smart_int(s, positive_only=False):
    i = 0
    v = None
    while i < len(s):
        try:
            if positive_only and s[i] == '-':
                raise ValueError
            v = int(s[i:], 10)
            break
        except ValueError:
            pass
        i += 1
    if v is None:
        raise ValueError("Cannot convert '%s' to int" % s)
    return v
    

class Station(object):
    """
    Class to represent the OVRO-LWA station and its antennas.
    """
    
    def __init__(self, name, lat, lon, elev, antennas=None):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.elev = elev
        
        self.antennas = []
        if antennas is not None:
            self.antennas = antennas
            
    @classmethod
    def from_line(cls, line):
        """
        Create a new Station instance from a line in an antenna positions file.
        """
        
        name, style, lat, lon, elev, x, y, active = line.split(None, 7)
        lat = float(lat) * numpy.pi/180
        lon = float(lon) * numpy.pi/180
        elev = 1182.89  ## Mean of the first 256 antennas
        return cls(name, lat, lon, elev)
        
    def append(self, ant):
        """
        Add an antenna to the array.
        """
        
        if not isinstance(ant, Antenna):
            raise TypeError("Expected an antenna")
        ant.parent = weakref.proxy(self)
        self.antennas.append(ant)
        
    def select_subset(self, ids):
        """
        Given a list of antenna IDs (either as integer index or name), return a
        new Station instance that only contains those antennas.
        """
        
        subset = Station(self.name+"-fast", self.lat*1.0, self.lon*1.0, self.elev*1.0)
        
        all_ids = [ant.id for ant in self.antennas]
        for id in ids:
            if isinstance(id, int):
                subset.append(copy.deepcopy(self.antennas[id]))
            else:
                subset.append(copy.deepcopy(self.antennas[all_ids.index(id)]))
        return subset
        
    @property
    def ecef(self):
        """
        Return the Earth centered, Earth fixed location of the array in meters.
        """
        
        e =  EarthLocation(lat=self.lat*u.rad, lon=self.lon*u.rad, height=self.elev*u.m)
        return (e.x.to_value(u.m), e.y.to_value(u.m), e.z.to_value(u.m))
        
    @property
    def topo_rot_matrix(self):
        """
        Return the rotation matrix that takes a difference in an Earth centered,
        Earth fixed location relative to the Station and rotates it into a
        topocentric frame that is south-east-zenith.
        """
        
        r = numpy.array([[ numpy.sin(self.lat)*numpy.cos(self.lon), numpy.sin(self.lat)*numpy.sin(self.lon), -numpy.cos(self.lat)],
                         [-numpy.sin(self.lon),                     numpy.cos(self.lon),                      0                  ],
                         [ numpy.cos(self.lat)*numpy.cos(self.lon), numpy.cos(self.lat)*numpy.sin(self.lon),  numpy.sin(self.lat)]])
        return r
        
    @property
    def casa_position(self):
        """
        Return a four-element tuple of (CASA position reference, CASA position 1,
        CASA position 2, CASA position 3, CASA position 4) that is suitable for
        use with casacore.measures.measures.position.
        """
        
        x, y, z = self.ecef
        return 'ITRF', '%fm' % x, '%fm' % y, '%fm' % z


class Antenna(object):
    """
    Class to represent an antenna in the OVRO-LWA.
    """
    
    def __init__(self, id, lat, lon, elev):
        if isinstance(id, str):
            id = _smart_int(id, positive_only=True)
        self.id = id
        self.lat = lat
        self.lon = lon
        self.elev = elev
        self.parent = None
        
    @classmethod
    def from_line(cls, line):
        """
        Create a new Antenna instance from a line in an antenna positions file.
        """
        
        name, style, lat, lon, elev, x, y, active = line.split(None, 7)
        lat = float(lat) * numpy.pi/180
        lon = float(lon) * numpy.pi/180
        try:
            elev = float(elev)
        except ValueError:
            elev = 1182.89  ## Mean of the first 256 antennas
        return cls(name, lat, lon, elev)
        
    @property
    def ecef(self):
        """
        Return the Earth centered, Earth fixed location of the antenna in meters.
        """
        
        e = EarthLocation(lat=self.lat*u.rad, lon=self.lon*u.rad, height=self.elev*u.m)
        return (e.x.to_value(u.m), e.y.to_value(u.m), e.z.to_value(u.m))
        
    @property
    def enz(self):
        """
        Return the topocentric east-north-zenith coordinates for the antenna 
        relative to the center of its associated Station in meters.
        """
        
        if self.parent is None:
            raise RuntimeError("Cannot find east-north-zenith without an associated Station")
            
        ecefFrom = numpy.array(self.parent.ecef)
        ecefTo = numpy.array(self.ecef)

        rho = ecefTo - ecefFrom
        rot = self.parent.topo_rot_matrix
        sez = numpy.dot(rot, rho)

        # Convert from south, east, zenith to east, north, zenith
        enz = 1.0*sez[[1,0,2]]
        enz[1] *= -1.0
        return enz


def parse_config(filename):
    """
    Given an OVRO-LWA configuration file, parse it and return a Station instance.
    """
    
    with open(filename, 'r') as fh:
        for line in fh:
            if len(line) < 3:
                continue
            elif line[0] == '#':
                continue
                
            if line.startswith('LWA-000'):
                line = line.replace('LWA-000', 'OVRO-LWA')
                station = Station.from_line(line)
            elif line.find('NO') > -2:
                ant = Antenna.from_line(line)
                station.append(ant)
                
    return station


# A ready-made Station instance, filled with Antennas
ovro = parse_config(OVRO_CONFIG_FILENAME)

# Use OVRO_MMA as the telescope name until CASA knows about OVRO-LWA
ovro.name = 'OVRO_MMA'

# Change the order to match what's going on in E2E
## The current list as of 2021 Sep 8
interim = ['LWA-213', 'LWA-224', 'LWA-215', 'LWA-214', 'LWA-218', 'LWA-217',
           'LWA-223', 'LWA-219', 'LWA-226', 'LWA-225', 'LWA-228', 'LWA-227',
           'LWA-230', 'LWA-229', 'LWA-232', 'LWA-231', 'LWA-234', 'LWA-233',
           'LWA-236', 'LWA-235', 'LWA-238', 'LWA-237', 'LWA-240', 'LWA-239',
           'LWA-242', 'LWA-241', 'LWA-244', 'LWA-243', 'LWA-246', 'LWA-245',
           'LWA-248', 'LWA-247']
interm = [_smart_int(v.replace('-', '')) for v in interim]
## Sort by swapping until there is nothing left to swap
while True:
    orig_order = [ant.id for ant in ovro.antennas]
    done = True
    for i,j in enumerate(interm):
        k = orig_order.index(j)
        if i != k:
            temp = ovro.antennas[k]
            ovro.antennas[k] = ovro.antennas[i]
            ovro.antennas[i] = temp
            done = False
            break
    if done:
        break
## Trim
ovro.antennas = ovro.antennas[:352]
         
