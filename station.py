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

# Change the order to match what's going on in Phase I
## The current list as of 2021 Nov 17
interim = ['LWA-001', 'LWA-002', 'LWA-003', 'LWA-004', 'LWA-005', 'LWA-006',
           'LWA-007', 'LWA-008', 'LWA-009', 'LWA-010', 'LWA-011', 'LWA-012',
           'LWA-013', 'LWA-014', 'LWA-015', 'LWA-016', 'LWA-017', 'LWA-018',
           'LWA-019', 'LWA-020', 'LWA-021', 'LWA-022', 'LWA-023', 'LWA-024',
           'LWA-025', 'LWA-026', 'LWA-027', 'LWA-028', 'LWA-029', 'LWA-030',
           'LWA-031', 'LWA-032', 'LWA-033', 'LWA-034', 'LWA-035', 'LWA-036',
           'LWA-037', 'LWA-038', 'LWA-039', 'LWA-040', 'LWA-041', 'LWA-042',
           'LWA-043', 'LWA-044', 'LWA-045', 'LWA-046', 'LWA-047', 'LWA-048',
           'LWA-049', 'LWA-050', 'LWA-051', 'LWA-052', 'LWA-053', 'LWA-054',
           'LWA-055', 'LWA-056', 'LWA-057', 'LWA-058', 'LWA-059', 'LWA-060',
           'LWA-061', 'LWA-062', 'LWA-063', 'LWA-064', 'LWA-065', 'LWA-066',
           'LWA-067', 'LWA-068', 'LWA-069', 'LWA-070', 'LWA-071', 'LWA-072',
           'LWA-073', 'LWA-074', 'LWA-075', 'LWA-076', 'LWA-077', 'LWA-078',
           'LWA-079', 'LWA-080', 'LWA-081', 'LWA-082', 'LWA-083', 'LWA-084', 
           'LWA-085', 'LWA-086', 'LWA-087', 'LWA-088', 'LWA-089', 'LWA-090',
           'LWA-091', 'LWA-092', 'LWA-093', 'LWA-094', 'LWA-095', 'LWA-096',
           'LWA-097', 'LWA-098', 'LWA-099', 'LWA-100', 'LWA-101', 'LWA-102',
           'LWA-103', 'LWA-104', 'LWA-105', 'LWA-106', 'LWA-107', 'LWA-108',
           'LWA-109', 'LWA-110', 'LWA-111', 'LWA-112', 'LWA-113', 'LWA-114',
           'LWA-115', 'LWA-116', 'LWA-117', 'LWA-118', 'LWA-119', 'LWA-120',
           'LWA-121', 'LWA-122', 'LWA-123', 'LWA-124', 'LWA-125', 'LWA-126',
           'LWA-127', 'LWA-128', 'LWA-129', 'LWA-130', 'LWA-131', 'LWA-132',
           'LWA-134', 'LWA-139', 'LWA-142', 'LWA-143', 'LWA-144', 'LWA-145',
           'LWA-146', 'LWA-147', 'LWA-148', 'LWA-149', 'LWA-150', 'LWA-151',
           'LWA-152', 'LWA-153', 'LWA-154', 'LWA-155', 'LWA-156', 'LWA-157',
           'LWA-158', 'LWA-159', 'LWA-160', 'LWA-168', 'LWA-169', 'LWA-172',
           'LWA-174', 'LWA-178', 'LWA-182', 'LWA-184', 'LWA-185', 'LWA-186',
           'LWA-187', 'LWA-188', 'LWA-189', 'LWA-190', 'LWA-191', 'LWA-192',
           'LWA-198', 'LWA-216', 'LWA-220', 'LWA-252', 'LWA-255', 'LWA-257',
           'LWA-258', 'LWA-259', 'LWA-260', 'LWA-261', 'LWA-262', 'LWA-263',
           'LWA-264', 'LWA-265', 'LWA-266', 'LWA-267', 'LWA-268', 'LWA-269',
           'LWA-270', 'LWA-271', 'LWA-272', 'LWA-273', 'LWA-274', 'LWA-275',
           'LWA-276', 'LWA-277', 'LWA-278', 'LWA-279', 'LWA-280', 'LWA-281',
           'LWA-282', 'LWA-283', 'LWA-284', 'LWA-285', 'LWA-286', 'LWA-287',
           'LWA-288', 'LWA-289', 'LWA-290', 'LWA-291', 'LWA-292', 'LWA-293',
           'LWA-294', 'LWA-295', 'LWA-296', 'LWA-297', 'LWA-298', 'LWA-299',
           'LWA-222', 'LWA-224', 'LWA-300', 'LWA-301', 'LWA-302', 'LWA-303',
           'LWA-304', 'LWA-305', 'LWA-133', 'LWA-135', 'LWA-136', 'LWA-137',
           'LWA-138', 'LWA-140', 'LWA-141', 'LWA-161', 'LWA-162', 'LWA-163',
           'LWA-164', 'LWA-165', 'LWA-166', 'LWA-167', 'LWA-226', 'LWA-193',
           'LWA-194', 'LWA-195', 'LWA-196', 'LWA-197', 'LWA-199', 'LWA-200',
           'LWA-201', 'LWA-202', 'LWA-225', 'LWA-227', 'LWA-253', 'LWA-306',
           'LWA-307', 'LWA-308', 'LWA-309', 'LWA-310', 'LWA-234', 'LWA-170',
           'LWA-171', 'LWA-173', 'LWA-175', 'LWA-176', 'LWA-203', 'LWA-204',
           'LWA-205', 'LWA-206', 'LWA-207', 'LWA-208', 'LWA-209', 'LWA-210',
           'LWA-230', 'LWA-228', 'LWA-229', 'LWA-231', 'LWA-232', 'LWA-233',
           'LWA-235', 'LWA-236', 'LWA-237', 'LWA-238', 'LWA-239', 'LWA-240',
           'LWA-254', 'LWA-311', 'LWA-312', 'LWA-313', 'LWA-314', 'LWA-315',
           'LWA-247', 'LWA-177', 'LWA-179', 'LWA-180', 'LWA-181', 'LWA-183',
           'LWA-211', 'LWA-212', 'LWA-213', 'LWA-214', 'LWA-215', 'LWA-243',
           'LWA-217', 'LWA-218', 'LWA-219', 'LWA-221', 'LWA-223', 'LWA-241',
           'LWA-242', 'LWA-244', 'LWA-245', 'LWA-246', 'LWA-248', 'LWA-249',
           'LWA-250', 'LWA-251', 'LWA-256', 'LWA-316', 'LWA-317', 'LWA-318',
           'LWA-319', 'LWA-320', 'LWA-321', 'LWA-322', 'LWA-323', 'LWA-324',
           'LWA-325', 'LWA-326', 'LWA-327', 'LWA-328', 'LWA-329', 'LWA-330',
           'LWA-331', 'LWA-332', 'LWA-333', 'LWA-334', 'LWA-335', 'LWA-336',
           'LWA-337', 'LWA-338', 'LWA-339', 'LWA-340', 'LWA-341', 'LWA-342',
           'LWA-343', 'LWA-344', 'LWA-345', 'LWA-346', 'LWA-347', 'LWA-348',
           'LWA-349', 'LWA-350', 'LWA-351', 'LWA-352']
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
         
