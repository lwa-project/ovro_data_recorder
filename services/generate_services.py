#!/usr/bin/env python3

import os
import sys
import glob
import json
import jinja2
import argparse
from hashlib import md5
from datetime import datetime

# Setup
## Path information
path = os.path.dirname(sys.executable)


def main(args):
    # Load in the configuration
    with open(args.config, 'r') as fh:
        config = json.loads(fh.read())
        
    # Generate the configuration tracking
    generated = datetime.utcnow().strftime("%Y/%m/%d %H:%M:%S UTC")
    input_file = os.path.abspath(args.config)
    with open(args.config, 'rb') as fh:
        md5sum = md5(fh.read())
        input_file_md5 = md5sum.hexdigest()
        
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
            for beam in config['power_beams'].keys():
                address   = config['power_beams'][beam]['ip']
                port      = config['power_beams'][beam]['port']
                directory = config['power_beams'][beam]['directory']
                quota     = config['power_beams'][beam]['quota']
                try:
                    logdir = config['power_beams'][beam]['logdir']
                except KeyError:
                    logdir = os.path.join(os.path.dirname(directory), 'log')
                if address != last_address:
                    cores = [32,33,34,35]
                    last_address = address
                service = template.render(path=path, anaconda=anaconda, condaenv=condaenv,
                                          beam=beam, address=address, port=port,
                                          directory=directory, quota=quota, logdir=logdir,
                                          cores=','.join([str(v) for v in cores]),
                                          generated=generated, input_file=input_file, input_file_md5=input_file_md5)
                for c in range(len(cores)):
                    cores[c] += len(cores)
                    cores[c] %= 48
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
            for band in config['slow_vis'].keys():
                address   = config['slow_vis'][band]['ip']
                port      = config['slow_vis'][band]['port']
                directory = config['slow_vis'][band]['directory']
                quota     = config['slow_vis'][band]['quota']
                try:
                    logdir = config['slow_vis'][band]['logdir']
                except KeyError:
                    logdir    = os.path.join(os.path.dirname(directory), 'log')
                if address != last_address:
                    cores = [40,41,42,43,44,45]
                    last_address = address
                service = template.render(path=path, anaconda=anaconda, condaenv=condaenv,
                                          band=band, address=address, port=port,
                                          directory=directory, quota=quota, logdir=logdir,
                                          cores=','.join([str(v) for v in cores]),
                                          generated=generated, input_file=input_file, input_file_md5=input_file_md5)
                for c in range(len(cores)):
                    cores[c] += len(cores)
                    cores[c] %= 48
                with open('dr-vslow-%s.service' % band, 'w') as fh:
                    fh.write(service)

            ### Manager
            template = env.get_template('dr-manager-vslow-base.service')
            band_id = []
            for band in config['slow_vis'].keys():
                address   = config['slow_vis'][band]['ip']
                port      = config['slow_vis'][band]['port']
                directory = config['slow_vis'][band]['directory']
                quota     = config['slow_vis'][band]['quota']
                base_ip = int(address.split('.')[-1], 10)
                base_port = port % 100
                band_id.append(str(base_ip*100 + base_port))
            band_id = ','.join(band_id)
            service = template.render(path=path, anaconda=anaconda, condaenv=condaenv,
                                      band_id=band_id, logdir=config['manager']['fast_vis']['logdir'],
                                      generated=generated, input_file=input_file, input_file_md5=input_file_md5)
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
            for band in config['fast_vis'].keys():
                address   = config['fast_vis'][band]['ip']
                port      = config['fast_vis'][band]['port']
                directory = config['fast_vis'][band]['directory']
                quota     = config['fast_vis'][band]['quota']
                try:
                    logdir = config['fast_vis'][band]['logdir']
                except KeyError:
                    logdir    = os.path.join(os.path.dirname(directory), 'log')
                if address != last_address:
                    cores = [42,43,44,45,46,47]
                    last_address = address
                service = template.render(path=path, anaconda=anaconda, condaenv=condaenv,
                                          band=band, address=address, port=port,
                                          directory=directory, quota=quota, logdir=logdir,
                                          cores=','.join([str(v) for v in cores]),
                                          generated=generated, input_file=input_file, input_file_md5=input_file_md5)
                for c in range(len(cores)):
                    cores[c] += len(cores)
                    cores[c] %= 48
                with open('dr-vfast-%s.service' % band, 'w') as fh:
                    fh.write(service)
                    
            ### Manager
            template = env.get_template('dr-manager-vfast-base.service')
            band_id = []
            for band in config['fast_vis'].keys():
                address   = config['fast_vis'][band]['ip']
                port      = config['fast_vis'][band]['port']
                directory = config['fast_vis'][band]['directory']
                quota     = config['fast_vis'][band]['quota']
                base_ip = int(address.split('.')[-1], 10)
                base_port = port % 100
                band_id.append(str(base_ip*100 + base_port))
            band_id = ','.join(band_id)
            service = template.render(path=path, anaconda=anaconda, condaenv=condaenv,
                                      band_id=band_id, logdir=config['manager']['fast_vis']['logdir'],
                                      generated=generated, input_file=input_file, input_file_md5=input_file_md5)
            with open('dr-manager-vfast.service', 'w') as fh:
                fh.write(service)
                
    ## T-engines
    if args.t_engines:
        if args.clean:
            filenames = glob.glob('./dr-tengine.service')
            filenames.extend(glob.glob('./dr-vbeam.service'))
            for filename in filenames:
                os.unlink(filename)
        else:
            template = env.get_template('dr-tengine-base.service')
            for beam in config['voltage_beams'].keys():
                address   = config['voltage_beams'][beam]['ip']
                port      = config['voltage_beams'][beam]['port']
                directory = config['voltage_beams'][beam]['directory']
                quota     = config['voltage_beams'][beam]['quota']
                try:
                    logdir = config['voltage_beams'][beam]['logdir']
                except KeyError:
                    logdir    = os.path.join(os.path.dirname(directory), 'log')
                service = template.render(path=path, anaconda=anaconda, condaenv=condaenv,
                                          address=address, port=port,
                                          directory=directory, quota=quota, logdir=logdir,
                                          generated=generated, input_file=input_file, input_file_md5=input_file_md5)
                with open('dr-tengine.service', 'w') as fh:
                    fh.write(service)
                    
            template = env.get_template('dr-vbeam-base.service')
            for beam in config['voltage_beams'].keys():
                address   = config['voltage_beams'][beam]['ip']
                port      = config['voltage_beams'][beam]['port']
                directory = config['voltage_beams'][beam]['directory'].replace('beam', 'vbeam')
                quota     = config['voltage_beams'][beam]['quota']
                try:
                    logdir = config['voltage_beams'][beam]['logdir']
                except KeyError:
                    logdir    = os.path.join(os.path.dirname(directory), 'log')
                service = template.render(path=path, anaconda=anaconda, condaenv=condaenv,
                                          address=address, port=port,
                                          directory=directory, quota=quota, logdir=logdir,
                                          generated=generated, input_file=input_file, input_file_md5=input_file_md5)
                with open('dr-vbeam.service', 'w') as fh:
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
    parser.add_argument('--config', type=str, default='service_config.json',
                       help='JSON file that specifies the data recorder mappings')
    parser.add_argument('-p', '--anaconda-path', type=str, default='/opt/miniconda3',
                       help='root path to anaconda install to use')
    parser.add_argument('-e', '--conda-env', type=str, default='datarecorder',
                       help='anaconda enviroment name to use')
    group = parser.add_mutually_exclusive_group(required=False)
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
    
