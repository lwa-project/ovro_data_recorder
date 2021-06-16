#!/usr/bin/env python

import os
import sys
import jinja2

# Setup
## Power beam setup
rdir = '/home/ubuntu/data/beam'
beams = { 1: ('enp216s0', 10000, rdir+'01'),
          2: ('enp216s0', 10001, rdir+'02'),
          3: ('enp216s0', 10002, rdir+'03'),
          4: ('enp216s0', 10003, rdir+'04'),
          5: ('enp216s0', 10004, rdir+'05'),
          6: ('enp216s0', 10005, rdir+'06'),
          7: ('enp216s0', 10006, rdir+'07'),
          8: ('enp216s0', 10007, rdir+'08'),
          9: ('enp216s0', 10008, rdir+'09'),
         10: ('enp216s0', 10009, rdir+'10'),
         11: ('enp216s0', 10010, rdir+'11'),
         12: ('enp216s0', 10011, rdir+'12'),
        }

## Slow visibilities setup
rdir = '/home/ubuntu/data/slow'
quota = 0
vslow = { 1: ('10.41.0.17', 10000, rdir, quota),
          2: ('10.41.0.18', 10000, rdir, quota),
          3: ('10.41.0.19', 10000, rdir, quota),
          4: ('10.41.0.20', 10000, rdir, quota),
          5: ('10.41.0.21', 10000, rdir, quota),
          6: ('10.41.0.22', 10000, rdir, quota),
          7: ('10.41.0.23', 10000, rdir, quota),
          8: ('10.41.0.24', 10000, rdir, quota),
          9: ('10.41.0.33', 10000, rdir, quota),
         10: ('10.41.0.34', 10000, rdir, quota),
         11: ('10.41.0.35', 10000, rdir, quota),
         12: ('10.41.0.36', 10000, rdir, quota),
         13: ('10.41.0.37', 10000, rdir, quota),
         14: ('10.41.0.38', 10000, rdir, quota),
         15: ('10.41.0.39', 10000, rdir, quota),
         16: ('10.41.0.40', 10000, rdir, quota),
        }

## Fast visibilities setup
rdir = '/home/ubuntu/data/fast'
quota = 10*1024**4
vfast = { 1: ('10.41.0.17', 11000, rdir, quota),
          2: ('10.41.0.18', 11000, rdir, quota),
          3: ('10.41.0.19', 11000, rdir, quota),
          4: ('10.41.0.20', 11000, rdir, quota),
          5: ('10.41.0.21', 11000, rdir, quota),
          6: ('10.41.0.22', 11000, rdir, quota),
          7: ('10.41.0.23', 11000, rdir, quota),
          8: ('10.41.0.24', 11000, rdir, quota),
          9: ('10.41.0.33', 11000, rdir, quota),
         10: ('10.41.0.34', 11000, rdir, quota),
         11: ('10.41.0.35', 11000, rdir, quota),
         12: ('10.41.0.36', 11000, rdir, quota),
         13: ('10.41.0.37', 11000, rdir, quota),
         14: ('10.41.0.38', 11000, rdir, quota),
         15: ('10.41.0.39', 11000, rdir, quota),
         16: ('10.41.0.40', 11000, rdir, quota),
        }

# Render
loader = jinja2.FileSystemLoader(searchpath='./')
env = jinja2.Environment(loader=loader)

## Power beams
template = env.get_template('dr-beam-base.service')
for beam in beams:
    address, port, directory = beams[beam] 
    service = template.render(beam=beam, address=address, port=port, directory=directory)
    with open('dr-beam-%s.service' % beam, 'w') as fh:
        fh.write(service)

## Slow visibilities
### Recorders
template = env.get_template('dr-vslow-base.service')
for band in vslow:
    address, port, directory, quota = vslow[band]
    service = template.render(band=band, address=address, port=port, directory=directory, quota=quota)
    with open('dr-vslow-%s.service' % band, 'w') as fh:
        fh.write(service)

### Manager
template = env.get_template('dr-manager-vslow-base.service')
begin_address = min([vslow[band][0] for band in vslow])
end_address = max([vslow[band][0] for band in vslow])
service = template.render(begin_address=begin_address, end_address=end_address)
with open('dr-manager-vslow.service', 'w') as fh:
    fh.write(service)

## Fast visibilities
### Recorders
emplate = env.get_template('dr-vfast-base.service')
for band in vfast:
    address, port, directory, quota = vfast[band]
    service = template.render(band=band, address=address, port=port, directory=directory, quota=quota)
    with open('dr-vfast-%s.service' % band, 'w') as fh:
        fh.write(service)
        
### Manager
template = env.get_template('dr-manager-vfast-base.service')
begin_address = min([vfast[band][0] for band in vslow])
end_address = max([vfast[band][0] for band in vslow])
service = template.render(begin_address=begin_address, end_address=end_address)
with open('dr-manager-vfast.service', 'w') as fh:
    fh.write(service)
