Data Recorder Pipelines for the OVRO-LWA Station
================================================

[![Documentation Status](https://readthedocs.org/projects/ovro-data-recorder/badge/?version=latest)](https://ovro-data-recorder.readthedocs.io/en/latest/?badge=latest)

DESCRIPTION
-----------
Bifrost-based data recorder pipelines for the power beams (`dr_beam.py`), correlator data (`dr_visibilities.py`),
and voltage beams (`dr_tengine.py`) data modes at the OVRO-LWA station.

REQUIREMENTS
------------
 * python >= 3.6
 * bifrost >= 0.9.0 + [OVRO-LWA data format support](https://github.com/realtimeradio/caltech-bifrost-dsp)
 * h5py
 * numpy
 * casacore
 * astropy
 * etcd3
 * pillow
 * dsa110-pyutils from https://github.com/dsa110/dsa110-pyutils
 * mnc_python from https://github.com/ovro-lwa/mnc_python
 * lwa_antpos from https://github.com/ovro-lwa/lwa-antpos
 * FFTW3 - single precision version
 * tar

INSTALLING
----------
No installation is necessary, just install the dependencies and launch the
pipelines.
