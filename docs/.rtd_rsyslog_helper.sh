#!/bin/bash

sudo bash -ec "echo -e \"module(load='imudp')\ninput(type='imudp' port='514')\" >> /etc/rsyslog.conf"
sudo restart rsyslog
