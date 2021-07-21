Data Recorder Pipelines for the OVRO-LWA Station
================================================

DESCRIPTION
-----------
Bifrost-based data recorder pipelines for the power beams (`dr_beam.py`), correlator data (`dr_visibilities.py`),
and voltage beams (`dr_tengine.py`) data modes at the OVRO-LWA station.

REQUIREMENTS
------------
 * python >= 2.7
 * bifrost >= 0.9.0 + [OVRO-LWA data format support](https://github.com/realtimeradio/caltech-bifrost-dsp)
 * h5py
 * numpy
 * casacore
 * astropy
 * etcd3
 * pillow
 * tar

INSTALLING
----------
No installation is necessary, just install the dependencies and launch the
pipelines.
