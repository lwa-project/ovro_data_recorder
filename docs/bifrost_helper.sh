#!/bin/bash

ROOT_PATH=`dirname $0`
cd ${ROOT_PATH}/../bifrost
./configure
make -j all
make install
