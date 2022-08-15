#!/bin/bash

# Hostname to subband/beam conversion
HOST=`hostname | grep lwacalim`
if [[ ${HOST} == "" ]]; then
  echo "Must be run on a calim node"
  exit 1
fi
HOST=`echo ${HOST} | sed -e 's/lwacalim//g;'`
HOST=`echo "10#${HOST}"`
SUB0=$(((${HOST}-1)*2+1))
SUB1=$(((${HOST}-1)*2+2))
if (( ${SUB0} > 16 )); then
  echo "No sub-bands or beams are recorded on this node"
  exit 1
fi

# Valid services
units=""
## Visibilities
for type in "slow" "fast"; do
  for SUB in ${SUB0} ${SUB1}; do
    units=`echo "${units} dr-v${type}-${SUB}"`
  done
done
## Beams
for BEAM in ${SUB0} ${SUB1}; do
  if (( ${BEAM} <= 12 )); then
    units=`echo "${units} dr-beam-${BEAM}"`
  fi
done
## Visibility managers
if (( ${HOST} == 1 )); then
  for type in "slow" "fast"; do
    units=`echo "${units} dr-manager-v${type}"`
  done
fi

# Stop all running
for unit in ${units}; do
  unit=`basename ${unit}`
  running=`systemctl --user list-units | grep ${unit} | grep running`
  if [[ "${running}" != "" ]]; then
    echo "Stopping '${unit}'..."
    systemctl --user stop ${unit}
  fi
done
