#!/bin/bash

BASEDIR=`realpath $0 | xargs dirname`
HOST=`hostname | sed -e 's/lwacalim//g;'`

cd ${BASEDIR}
python3 ./generate_services.py

RELOAD=0
if [[ "${HOST}" == "01" ]]; then
  for SUB in 1 2; do
    cp -v dr-v*-${SUB}.service /etc/systemd/system/
    cp -v dr-beam-${SUB}.service /etc/systemd/system/
    cp -v dr-manager-vslow.service /etc/systemd/system/
    cp -v dr-manager-vfast.service /etc/systemd/system/
  done
  RELOAD=1
fi

if [[ "${HOST}" == "02" ]]; then
  for SUB in 3 4; do
    cp -v dr-v*-${SUB}.service /etc/systemd/system/
    cp -v dr-beam-${SUB}.service /etc/systemd/system/
  done
  RELOAD=1
fi

if [[ "${HOST}" == "03" ]]; then
  for SUB in 5 6; do
    cp -v dr-v*-${SUB}.service /etc/systemd/system/
    cp -v dr-beam-${SUB}.service /etc/systemd/system/
  done
  RELOAD=1
fi

if [[ "${HOST}" == "04" ]]; then
  for SUB in 7 8; do
    cp -v dr-v*-${SUB}.service /etc/systemd/system/
    cp -v dr-beam-${SUB}.service /etc/systemd/system/
  done
  RELOAD=1
fi

if [[ "${HOST}" == "05" ]]; then
  for SUB in 9 10; do
    cp -v dr-v*-${SUB}.service /etc/systemd/system/
    cp -v dr-beam-${SUB}.service /etc/systemd/system/
  done
  RELOAD=1
fi

if [[ "${HOST}" == "06" ]]; then
  for SUB in 11 12; do
    cp -v dr-v*-${SUB}.service /etc/systemd/system/
    cp -v dr-beam-${SUB}.service /etc/systemd/system/
  done
  RELOAD=1
fi

if [[ "${HOST}" == "07" ]]; then
  for SUB in 13 14; do
    cp -v dr-v*-${SUB}.service /etc/systemd/system/
    #cp -v dr-beam-${SUB}.service /etc/systemd/system/
  done
  RELOAD=1
fi

if [[ "${HOST}" == "08" ]]; then
  for SUB in 15 16; do
    cp -v dr-v*-${SUB}.service /etc/systemd/system/
    #cp -v dr-beam-${SUB}.service /etc/systemd/system/
  done
  RELOAD=1
fi

if [[ "${RELOAD}" == "1" ]]; then
  systemctl daemon-reload
else
  echo "No services installed"
fi
