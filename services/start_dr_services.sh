#!/bin/bash

units=`ls -1 /etc/systemd/system/dr-*.service`
for unit in ${units}; do
  unit=`basename ${unit}`
  running=`systemctl list-units | grep ${unit} | grep running`
  if [[ "${running}" == "" ]]; then
    echo "Starting '${unit}'..."
    systemctl start ${unit}
  fi
done
