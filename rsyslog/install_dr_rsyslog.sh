#!/bin/bash

HOST=`hostname | grep lwacalim`
if [[ ${HOST} == "" ]]; then
  echo "Must be run on a calim node"
  exit 1
fi
cp *-drservices-${HOST}.conf /etc/rsyslog.d/

systemctl restart rsyslog.service
