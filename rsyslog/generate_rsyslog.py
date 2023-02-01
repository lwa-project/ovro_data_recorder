#!/usr/bin/env python3

import os
import sys
import glob
import argparse


def main(args):
    for node in range(1, 8+1):
        hostname = f"lwacalim{node:02d}"
        
        if args.clean:
            filenames = glob.glob(f"*-drservices-{hostname}.conf")
            for filename in filenames:
                os.unlink(filename)
            
        else:
            vis_idx = [2*(node-1)+i+1 for i in range(2)]
            beam_idx = [2*(node-1)+i+1 for i in range(2)]
            beam_idx = list(filter(lambda x: x<=12, beam_idx))
            
            with open(f"{args.priority}-drservices-{hostname}.conf", 'w') as fh:
                for i in vis_idx:
                    fh.write(f"""
:programname, isequal, "dr-vslow-{i}" /data{node:02d}/log/dr-vslow-{i}.{hostname}.syslog
& stop

:programname, isequal, "dr-vfast-{i}" /data{node:02d}/log/dr-vfast-{i}.{hostname}.syslog
& stop

""")
                    
                for i in beam_idx:
                    fh.write(f"""
:programname, isequal, "dr-beam-{i}" /data{node:02d}/log/dr-beam-{i}.{hostname}.syslog
& stop
""")
                    
                if node == 1:
                    fh.write(f"""
:programname, isequal, "dr-manager-vslow" /data{node:02d}/log/dr-manager-vslow.{hostname}.syslog
& stop

:programname, isequal, "dr-manager-vfast" /data{node:02d}/log/dr-manager-vfast.{hostname}.syslog
& stop
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='generate rsyslog configuration files for the data recorder pipelines', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument('-p', '--priority', type=int, default=10,
                        help='configuration file priority')
    parser.add_argument('-c', '--clean', action='store_true',
                        help='delete the generated configuration files')
    args = parser.parse_args()
    if args.priority < 1 or args.priority > 99:
        raise RuntimeError(f"Invalid priority value '{args.priority}'")
    main(args)
    
