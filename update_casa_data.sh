#!/bin/bash

if [[ -e /home/pipeline/.conda/envs/datarecorder ]]; then
  mkdir -p /home/pipeline/.conda/envs/datarecorder/lib/casa/data/
  cd /home/pipeline/.conda/envs/datarecorder/lib/casa/data/
  rsync -avz rsync://casa-rsync.nrao.edu/casa-data .
fi
