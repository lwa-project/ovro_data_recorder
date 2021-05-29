Requirements
============

The OVRO-LWA data recording pipelines are written in Python using the Bifrost
pipeline frame.  The following Python packages are needed to full the pipelines:

 * python >= 3.6
 * bifrost >= 0.9 - "disk-readers" branch
 * numpy >= 1.19.5
 * matplotlib >= 3.2.2
 * PIL >= 8.0.1
 * astropy >= 4.0.1.post1
 * casacore >= 3.0.0
 * h5py >= 2.7.1
 * etcd3 >= 0.12.0

Older versions of the packages listed above may work but have not been tested.

Installing
==========

Once the requirements are met no additional installation steps are required.  Simply
run the pipelines with the :doc:`desired command line arguments</using>`.
