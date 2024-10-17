#!/bin/bash

BASEDIR=`dirname $0 | xargs realpath`
HOST=`hostname | grep lwastor`
if [[ ${HOST} == "" ]]; then
  echo "Must be run on a lwastor node"
  exit 1
fi
HOST=`echo ${HOST} | sed -e 's/lwastor//g;'`
HOST=`echo "10#${HOST}"`
SUB0=$(((${HOST}-1)*2+1))
SUB1=$(((${HOST}-1)*2+2))
if (( ${SUB0} > 16 )); then
  echo "No sub-bands or beams are recorded on this node"
  exit 1
fi

cd ${BASEDIR}
./generate_services.py

mkdir -p /home/ubuntu/.config/systemd/user/

for SUB in `seq 1 1 16`; do
  if [[ "${SUB0}" == "${SUB} || [[ "${SUB1}" == "${SUB} ]]; then
    cp -v dr-v*-${SUB}.service /home/ubuntu/.config/systemd/user/
  fi
done
for BEAM in `seq 1 1 12`; do
  if [[ "${SUB0}" == "${BEAM} || [[ "${SUB1}" == "${BEAM} ]]; then
    cp -v dr-beam-${BEAM}.service /home/ubuntu/.config/systemd/user/
  fi
done
if (( ${HOST} == 1 )); then
  cp -v dr-manager-vslow.service /home/ubuntu/.config/systemd/user/
  cp -v dr-manager-vfast.service /home/ubuntu/.config/systemd/user/
fi

systemctl --user daemon-reload
