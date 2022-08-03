#!/bin/bash

units=`systemctl list-units | grep dr- | awk '{print $1}'`
for unit in ${units}; do
  echo "Restarting '${unit}'..."
  systemctl restart ${unit}
done
