#!/bin/bash

pip install setuptools_scm
pip install etcd3 \
  "git+https://github.com/dsa110/dsa110-pyutils@v3.8.2" \
  "git+https://github.com/ovro-lwa/mnc_python@0.8.12" \
  "git+https://github.com/ovro-lwa/lwa-antpos@0.6.10" \
  "git+https://github.com/ovro-lwa/lwa-pyutils@v1.4.5" \
  "git+https://github.com/ovro-lwa/lwa-observing@0.2.0" \
  pint \
  "git+https://github.com/realtimeradio/caltech-bifrost-dsp#subdirectory=pipeline-control"
