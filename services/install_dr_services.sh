#!/bin/bash

BASEDIR=`dirname $0 | xargs realpath`
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

cd ${BASEDIR}
./generate_services.py -a -e deployment

mkdir -p /home/pipeline/.config/systemd/user/

for SUB in `seq 1 1 16`; do
  cp -v dr-v*-${SUB}.service /home/pipeline/.config/systemd/user/
done
for BEAM in `seq 1 1 12`; do
  cp -v dr-beam-${BEAM}.service /home/pipeline/.config/systemd/user/
done
cp -v dr-manager-vslow.service /home/pipeline/.config/systemd/user/
cp -v dr-manager-vfast.service /home/pipeline/.config/systemd/user/

systemctl --user daemon-reload
