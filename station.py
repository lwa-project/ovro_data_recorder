import os
import sys

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
        name, lat, lng, x, y, active = line.split(None, 5)
        lat = float(lat)
        lng = float(lng)
        elev = 1222.0        # Is this right?
        return cls(name, lat, lng, elev)
        
    def append(self, ant):
        """
        Add an antenna to the array.
        """
        
        if not isinstance(ant, Antenna):
            raise TypeError("Expected an antenna")
        self.antennas.append(ant)


class Antenna(object):
    """
    Class to represent an antenna in the OVRO-LWA.
    """
    
    def __init__(self, id, x, y, z):
        if isinstance(id, str):
            id = smart_int(id)
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        
    @classmethod
    def from_line(cls, line):
        name, lat, lng, x, y, active = line.split(None, 5)
        x = float(x)
        y = float(y)
        z = 1.5     # They are about that tall
        return cls(name, x, y, z)
        
    
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
