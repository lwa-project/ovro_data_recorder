import os
import sys
import weakref

from astropy.coordinates import EarthLocation
import astropy.units as u


CONFIG_FILENAME = os.path.join(os.path.dirname(__file__), 'ovro.txt')


def smart_int(s):
    i = 0
    v = None
    while i < len(s):
        try:
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
        
        name, lat, lon, x, y, active = line.split(None, 5)
        lat = float(lat)
        lon = float(lon)
        elev = 1222.0        # Is this right?
        return cls(name, lat, lon, elev)
        
    def append(self, ant):
        """
        Add an antenna to the array.
        """
        
        if not isinstance(ant, Antenna):
            raise TypeError("Expected an antenna")
        ant.parent = weakref.proxy(self)
        self.antennas.append(ant)
        
    @property
    def astropy(self):
        """
        Return an AstroPy EarthLocation instance describing the array.
        """
        
        return  EarthLocation(lat=self.lat*u.deg, lon=self.lon*u.deg, height=self.elev*u.m)
        
    @property
    def ecef(self):
        """
        Return the Earth centered, Earth fixed location of the array in meters.
        """
        
        e = self.astropy
        return (e.x.to_value(u.m), e.y.to_value(u.m), e.z.to_value(u.m))


class Antenna(object):
    """
    Class to represent an antenna in the OVRO-LWA.
    """
    
    def __init__(self, id, lat, lon, elev):
        if isinstance(id, str):
            id = smart_int(id)
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
        
        name, lat, lon, x, y, active = line.split(None, 5)
        lat = float(lat)
        lon = float(lon)
        elev = 1222.0        # Is this right?
        return cls(name, lat, lon, elev)
        
    @property
    def astropy(self):
        """
        Return an AstroPy EarthLocation instance describing the antenna.
        """
        
        return  EarthLocation(lat=self.lat*u.deg, lon=self.lon*u.deg, height=self.elev*u.m)
        
    @property
    def ecef(self):
        """
        Return the Earth centered, Earth fixed location of the antenna in meters.
        """
        
        e = self.astropy
        return (e.x.to_value(u.m), e.y.to_value(u.m), e.z.to_value(u.m))
        
    
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
                station = Station.from_line(line)
            elif line.find('NO') == -1:
                ant = Antenna.from_line(line)
                station.append(ant)
                
    return station


# A ready-made Station instance, filled with Antennas
ovro = parse_config(CONFIG_FILENAME)
