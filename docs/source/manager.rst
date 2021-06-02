dr_manager_visibilities.py
==========================

To help manage the various fast and slow visibility pipelines there is
``dr_manager_visibilities.py``.  This script exposes a subset of the ``dr_visibilities.py``
command and monitoring points through a unified interface such that a single call
can be used to control and monitor all of the fast or slow recording pipelines.

Structure
---------

The manager is written as a single threaded application that listens for visibility
recording commands and then forwards those command on to all of the recording 
pipelines that it manages.  It then aggregates the responses and returns a single
dictionary that includes the responses of the pipelines.  Similarly, for monitoring
points, the manager aggregates the monitoring points of the individual pipelines. 

The manager is designed such that there is one manager per mode (fast and slow).
Thus, there will be two manager instances.

Control Commands
----------------

The ``dr_manager_visibilities.py`` supports four commands: ``ping``, ``sync``, ``start``, 
and ``stop``.
 
 * ``ping`` - This command simply replies which is helpful to see if the pipeline
   is responsive.  There are no required or optional arguments.  Returns a dictionary
   of the ping response from each managed pipeline.
 * ``sync`` - This command sets the system time via NTP.  The required arguments to
   this command are:
   
    * ``server`` - a NTP server name or IP address to sync against.
    
    There are no optional arguments.  The command returns a dictionary of the sync
    status from each managed pipeline.
 * ``start`` - This schedules when to start a recording.  The required arguments
   to this command are:
   
    * ``start_mjd`` - an integer MJD value for when the recording will start or
      "now" to start the recording 15 s after the command is received and
    * ``start_mpm`` - an integer number of milliseconds past midnight value on the
      MJD specified in ``start_mjd`` for when the recording will start.
    
  There are no optional arguments.  The command returns a dictionary of the base
  name for the files from each managed pipeline.
 * ``stop`` - This schedules when to stop a recording.  The required arguments to
   this command are:
   
    * `stop_mjd` - an integer MJD value for when the recording will stop or
      "now" to stop the recording 15 s after the command is received and
    * ``stop_mpm`` - an integer number of milliseconds past midnight value on the
      MJD specified in ``stop_mjd`` for when the recording will stop.
      
   There are no optional arguments.  The command returns a dictionary of the base
   name for the files from each managed pipeline associated with the stopped
   recording.

Monitoring Points
-----------------

There are several monitoring points for the pipeline.  All monitoring points live
under the etcd key "/mon/drv[sf]", where [sf] indicates slow (s) or fast (f) mode.

  * /mon/drv[sf]/summary - An overall status of the managed pipelines.  Possible values
    are "normal", "warning", and "error".
  * /mon/drv[sf]/info - A more detailed explanation of the summary condition as a
    a string that gives the individual summary values for all of the managed
    pipelines.
