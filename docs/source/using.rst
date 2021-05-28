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

Commanding the Pipelines
------------------------

You can send a command to a pipeline with the mcs.py module::
  
  from mcs import Client
  
  # Create an anonymous client
  c = Client()
  
  # Send the command
  response = c.send_command(subsystem_id, 'record',
                            start_mjd=mjd_now,
                            start_mpm=mpm_now,
                            duration_ms=30*1000)
  
  # Print the response
  print(response)
