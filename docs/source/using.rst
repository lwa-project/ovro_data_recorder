Running and Using the Recorders
===============================

Command Line Signatures
-----------------------

dr_visibilities.py
^^^^^^^^^^^^^^^^^^

.. include:: dr_visibilities.help

dr_manager_visibilities.py
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: dr_manager_visibilities.help

dr_beam.py
^^^^^^^^^^

.. include:: dr_beam.help

dr_tengine.py
^^^^^^^^^^^^^

.. include:: dr_tengine.help

Systemd User Services
---------------------

The various data recorder pipelines can be launched as systemd user services.  The
included `utils/generate_services.py` helps with defining these services:

.. include:: generate_services.help

These services are configured for the `lwacalim` cluster by default.

Interacting with the Pipelines
------------------------------

You can read monitoring points or send a command to a pipeline with the mnc.mcs
module::
  
  >>> from mnc.mcs import Client
  >>> 
  >>> # Create an anonymous client to talk to various subsystems
  >>> c = Client()
  >>> 
  >>> # Read the "bifrost/pipeline_lag" monitor point from the first power
  >>> # beam (dr1)
  >>> lag = c.read_monitor_point('bifrost/pipeline_lag', 'dr1')
  >>> print(lag)
  6.689624s at 2021-06-02 14:37:06.760481
  >>>
  >>> # Read a monitor point that does not exist
  >>> lag = c.read_monitor_point('bifrost/pipeline_', 'dr1')
  >>> print(lag)
  None
  >>>
  >>> # Send a "ping" command to the first power beam
  >>> response = c.send_command('dr1', 'ping')
  >>> print(response)
  (True, {'sequence_id': '12a3606ac3b011eb9eb410bf48e38102',
  'timestamp': 1622644655.9322925, 'status': 'success',
  'response': 'pong'})
  >>>
  >>> # Send a "ping" command to a subsystem that does not exist
  >>> response = c.send_command('a_cat', 'ping')
  >>> print(response)
  (False, '5d42c5acc3b011eb9eb410bf48e38102')
  >>>
  >>> # Send a command "record" command to the first power beam
  >>> from mnc.common import LWATime
  >>> t_now = LWATime.now()
  >>> mjd_now = int(t_now.mjd)
  >>> mpm_now = int((t_now.mjd - mjd_now)*86400.0*1000.0)
  >>> response = c.send_command('dr1', 'record',
                                start_mjd=mjd_now,
                                start_mpm=mpm_now,
                                duration_ms=30*1000)
  >>> print(response)
  (True, {'sequence_id': '76c6013ec3b411eb9eb410bf48e38102',
  'timestamp': 1622646541.9380054, 'status': 'success',
  'response': {'filename': '/home/jdowell/CodeSafe/ovro_data_recorder/059367_76c6013ec3b411eb9eb410bf48e38102'}})
