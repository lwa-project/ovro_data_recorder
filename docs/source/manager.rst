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

The ``dr_visibilities.py`` pipeline supports three commands: ``record``, ``cancel``, 
and ``delete``.

 * ``record`` - This schedules a recording to take place.  The required arguments to
   this command are:
   
    * ``start_mjd`` - an integer MJD value for when the recording will start,
    * ``start_mpm`` - an integer number of milliseconds past midnight value on the
      MJD specified in ``start_mjd`` for when the recording will start, and
    * ``duration_ms`` - the number of milliseconds to record data for.
    
  There are no optional arguments.  The command returns a dictionary of the base
  name for the files from each managed pipeline.
 * ``cancel`` - This cancels are previously scheduled or active recording.  The
    required arguments to this command are:
    
     * `queue_id` - an entry number in the recording queue to cancel.
     
   There are no optional arguments.  The command returns a dictionary of the base
   name for the files from each managed pipeline assoiated with the queue entry.

Monitoring Points
-----------------

There are several monitoring points for the pipeline.  All monitoring points live
under the etcd key "/mon/drv[sf]", where [sf] indicates slow (s) or fast (f) mode.

  * /mon/drv[sf]/summary - An overall status of the managed pipelines.  Possible values
    are "normal", "warning", and "error".
  * /mon/drv[sf]/info - A more detailed explanation of the summary condition as a
    a string that gives the individual summary values for all of the managed
    pipelines.
