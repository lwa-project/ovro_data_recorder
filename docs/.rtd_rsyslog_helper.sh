#!/bin/bash

echo -e 'module(load="imudp")\ninput(type="imudp" port="514")' >> /etc/rsyslog.conf
systemctl restart rsyslog
