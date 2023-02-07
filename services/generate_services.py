#!/usr/bin/env python

import os
import sys
import glob
import jinja2
import argparse

# Setup
## Path information
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

## Power beam setup
rdir = '/data{{ calim_host }}/beam'
quota = '500GB'
beams = { 1: ('10.41.0.76', 20001, rdir+'01', quota),
          2: ('10.41.0.76', 20002, rdir+'02', quota),
          3: ('10.41.0.77', 20001, rdir+'03', quota),
          4: ('10.41.0.77', 20002, rdir+'04', quota),
          5: ('10.41.0.78', 20001, rdir+'05', quota),
          6: ('10.41.0.78', 20002, rdir+'06', quota),
          7: ('10.41.0.79', 20001, rdir+'07', quota),
          8: ('10.41.0.79', 20002, rdir+'08', quota),
          9: ('10.41.0.80', 20001, rdir+'09', quota),
         10: ('10.41.0.80', 20002, rdir+'10', quota),
         11: ('10.41.0.81', 20001, rdir+'11', quota),
         12: ('10.41.0.81', 20001, rdir+'12', quota),
        }

## Slow visibilities setup
rdir = '/data{{ calim_host }}/slow'
quota = '8TB'
vslow = { 1: ('10.41.0.76', 10001, rdir, quota),
          2: ('10.41.0.76', 10002, rdir, 0),
          3: ('10.41.0.77', 10001, rdir, quota),
          4: ('10.41.0.77', 10002, rdir, 0),
          5: ('10.41.0.78', 10001, rdir, quota),
          6: ('10.41.0.78', 10002, rdir, 0),
          7: ('10.41.0.79', 10001, rdir, quota),
          8: ('10.41.0.79', 10002, rdir, 0),
          9: ('10.41.0.80', 10001, rdir, quota),
         10: ('10.41.0.80', 10002, rdir, 0),
         11: ('10.41.0.81', 10001, rdir, quota),
         12: ('10.41.0.81', 10002, rdir, 0),
         13: ('10.41.0.82', 10001, rdir, quota),
         14: ('10.41.0.82', 10002, rdir, 0),
         15: ('10.41.0.83', 10001, rdir, quota),
         16: ('10.41.0.83', 10002, rdir, 0),
        }

## Fast visibilities setup
rdir = '/data{{ calim_host }}/fast'
quota = '500GB'
vfast = { 1: ('10.41.0.76', 11001, rdir, quota),
          2: ('10.41.0.76', 11002, rdir, 0),
          3: ('10.41.0.77', 11001, rdir, quota),
          4: ('10.41.0.77', 11002, rdir, 0),
          5: ('10.41.0.78', 11001, rdir, quota),
          6: ('10.41.0.78', 11002, rdir, 0),
          7: ('10.41.0.79', 11001, rdir, quota),
          8: ('10.41.0.79', 11002, rdir, 0),
          9: ('10.41.0.80', 11001, rdir, quota),
         10: ('10.41.0.80', 11002, rdir, 0),
         11: ('10.41.0.81', 11001, rdir, quota),
         12: ('10.41.0.81', 11002, rdir, 0),
         13: ('10.41.0.82', 11001, rdir, quota),
         14: ('10.41.0.82', 11002, rdir, 0),
         15: ('10.41.0.83', 11001, rdir, quota),
         16: ('10.41.0.83', 11002, rdir, 0),
        }

## T-engine setup
rdir = '/home/ubuntu/data/tengine'
quota = 0
tengines = {1: ('10.41.0.73', 21001, rdir, quota),
           }

def main(args):
    # Pre-process
    anaconda = args.anaconda_path
    condaenv = args.conda_env
    if (not args.power_beams \
        and not args.slow_visibilities \
        and not args.fast_visibilities \
        and not args.t_engines):
       args.power_beams = args.slow_visibilities = args.fast_visibilities = args.t_engines = True
       
    # Render
    loader = jinja2.FileSystemLoader(searchpath='./')
    env = jinja2.Environment(loader=loader)

    ## Power beams
    if args.power_beams:
        if args.clean:
            filenames = glob.glob('./dr-beam-[0-9]*.service')
            for filename in filenames:
                os.unlink(filename)
        else:
            template = env.get_template('dr-beam-base.service')
            last_address = None
            for beam in beams:
                address, port, directory, quota = beams[beam]
                calim_host = '%02d' % (int(address.split('.', 3)[-1],10)-75,)
                if address != last_address:
                    cores = [72,73,74,75]
                    last_address = address
                directory = env.from_string(directory)
                directory = directory.render(calim_host=calim_host)
                service = template.render(path=path, anaconda=anaconda, condaenv=condaenv,
                                          beam=beam, address=address, port=port,
                                          calim_host=calim_host,
                                          directory=directory, quota=quota,
                                          cores=','.join([str(v) for v in cores]))
                for c in range(len(cores)):
                    cores[c] += len(cores)
                    cores[c] %= 96
                with open('dr-beam-%s.service' % beam, 'w') as fh:
                    fh.write(service)

    ## Slow visibilities
    if args.slow_visibilities:
        if args.clean:
            filenames = glob.glob('./dr-vslow-[0-9]*.service')
            filenames.extend(glob.glob('./dr-manager-vslow.service'))
            for filename in filenames:
                os.unlink(filename)
        else:
            ### Recorders
            template = env.get_template('dr-vslow-base.service')
            last_address = None
            for band in vslow:
                address, port, directory, quota = vslow[band]
                calim_host = '%02d' % (int(address.split('.', 3)[-1],10)-75,)
                if address != last_address:
                    cores = [50,51,52,53,54,55]
                    last_address = address
                directory = env.from_string(directory)
                directory = directory.render(calim_host=calim_host)
                service = template.render(path=path, anaconda=anaconda, condaenv=condaenv,
                                          band=band, address=address, port=port,
                                          calim_host=calim_host,
                                          directory=directory, quota=quota,
                                          cores=','.join([str(v) for v in cores]))
                for c in range(len(cores)):
                    cores[c] += len(cores)
                    cores[c] %= 96
                with open('dr-vslow-%s.service' % band, 'w') as fh:
                    fh.write(service)

            ### Manager
            template = env.get_template('dr-manager-vslow-base.service')
            band_id = []
            for band in vslow:
                address, port, directory, quota = vslow[band]
                base_ip = int(address.split('.')[-1], 10)
                base_port = port % 100
                band_id.append(str(base_ip*100 + base_port))
            band_id = ','.join(band_id)
            service = template.render(path=path, anaconda=anaconda, condaenv=condaenv,
                                      band_id=band_id)
            with open('dr-manager-vslow.service', 'w') as fh:
                fh.write(service)

    ## Fast visibilities
    if args.fast_visibilities:
        if args.clean:
            filenames = glob.glob('./dr-vfast-[0-9]*.service')
            filenames.extend(glob.glob('./dr-manager-vfast.service'))
            for filename in filenames:
                os.unlink(filename)
        else:
            ### Recorders
            template = env.get_template('dr-vfast-base.service')
            last_address = None
            for band in vfast:
                address, port, directory, quota = vfast[band]
                calim_host = '%02d' % (int(address.split('.', 3)[-1],10)-75,)
                if address != last_address:
                    cores = [62,63,64,65,66,67]
                    last_address = address
                directory = env.from_string(directory)
                directory = directory.render(calim_host=calim_host)
                service = template.render(path=path, anaconda=anaconda, condaenv=condaenv,
                                          band=band, address=address, port=port,
                                          calim_host=calim_host,
                                          directory=directory, quota=quota,
                                          cores=','.join([str(v) for v in cores]))
                for c in range(len(cores)):
                    cores[c] += len(cores)
                    cores[c] %= 96
                with open('dr-vfast-%s.service' % band, 'w') as fh:
                    fh.write(service)
                    
            ### Manager
            template = env.get_template('dr-manager-vfast-base.service')
            band_id = []
            for band in vfast:
                address, port, directory, quota = vfast[band]
                base_ip = int(address.split('.')[-1], 10)
                base_port = port % 100
                band_id.append(str(base_ip*100 + base_port))
            band_id = ','.join(band_id)
            service = template.render(path=path, anaconda=anaconda, condaenv=condaenv,
                                      band_id=band_id)
            with open('dr-manager-vfast.service', 'w') as fh:
                fh.write(service)
                
    ## T-engines
    if args.t_engines:
        if args.clean:
            filenames = glob.glob('./dr-tengine.service')
            for filename in filenames:
                os.unlink(filename)
        else:
            template = env.get_template('dr-tengine-base.service')
            for beam in tengines:
                address, port, directory, quota = tengines[beam]
                service = template.render(path=path, anaconda=anaconda, condaenv=condaenv,
                                          address=address, port=port,
                                          directory=directory, quota=quota)
                with open('dr-tengine.service', 'w') as fh:
                    fh.write(service)
                    
    if not args.clean:
        print("To enable/update these services:")
        print(" * copy the relevant .service files to /home/pipeline/.config/systemd/user/")
        print(" * reload systemd with 'systemctl --user daemon-reload'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='generate systemd service files for the data recorder pipelines', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-p', '--anaconda-path', type=str, default='/opt/miniconda3',
                       help='root path to anaconda install to use')
    group.add_argument('-e', '--conda-env', type=str, default='datarecorder',
                       help='anaconda enviroment name to use')
    group.add_argument('-b', '--power-beams', action='store_true',
                       help='only generate/clean the power beam services')
    group.add_argument('-s', '--slow-visibilities', action='store_true',
                       help='only generate/clean the slow visibitlies services')
    group.add_argument('-f', '--fast-visibilities', action='store_true',
                       help='only generate/clean the fast visibilities services')
    group.add_argument('-t', '--t-engines', action='store_true',
                       help='only generate/clean the T-engine services')
    group.add_argument('-a', '--all', action='store_false',
                       help='generate/clean all services')
    parser.add_argument('-c', '--clean', action='store_true',
                        help='delete the generated services')
    args = parser.parse_args()
    main(args)
    
