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
 * bifrost >= 0.9.0 + OVRO-LWA data format support (included with the `bifrost` submodule)
 * h5py
 * numpy
 * scipy
 * casacore
 * astropy
 * etcd3
 * pillow
 * jinja2
 * pyzmq
 * mnc_python from https://github.com/ovro-lwa/mnc_python
 * lwa_antpos from https://github.com/ovro-lwa/lwa-antpos
 * lwa_observing from https://github.com/ovro-lwa/lwa-observing
 * lwa352_pipeline_control from https://github.com/realtimeradio/caltech-bifrost-dsp
 * FFTW3 - single precision version
 * tar

INSTALLING
----------
Install ovro_data_recorder by running:

	pip install -e .

or by using the included `deploy` script.
