#!/bin/bash

units=`systemctl list-units | grep dr- | awk '{print $1}'`
for unit in ${units}; do
  echo "Stopping '${unit}'..."
  systemctl stop ${unit}
done
