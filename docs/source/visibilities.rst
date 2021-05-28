dr_visibilities.py
==================

``dr_visibilities.py`` is the recording pipeline for both the fast and slow
visibility modes.  This pipeline captures packetized X-engine data from the
digital system and writes the data to CASA measurement sets.  In addition to
recording data the pipeline also generates diagnostic plots of the data showing
the auto-correlation spectra as well as plots of visibility amplitude as a 
function of `(u,v)` radial distance.

Structure
---------

The pipeline is written in the Bifrost framework and has five blocks:  
``CaptureOp``, ``SpectraOp``, ``BaselineOp``, ``StatisticsOp``, and ``WriterOp``.

 * ``CaptureOp`` - This is the data capture block which is responsible for capturing
   the visibility packets from the digital system, ordering them in time and baseline,
   and writing the organized data to a Bifrost ring.
 * ``SpectraOp`` - This reads in the visibility data and writes a 352-frame PNG file
   every 60 s that contains the auto-correlation spectra for every antenna.  This
   block is only active for the slow visibility mode.
 * ``BaselineOp`` - This reads in the visibility data and writes a PNG file that shows
   the visibility amplitude for the center frequency channel as a function of `(u,v)`
   radial distance every 60 s.  This block is only active for the slow visibility
   mode.
 * ``StatisticsOp`` - This reads in the visibility data and computes per-antenna
   minimum/mean/maximum values across frequency from the auto-correlations on a 
   60 s cadence.
 * ``WriterOp`` - This reads in the visibility data and writes CASA measurement sets
   to disk.

The pipeline is designed such that there is one pipeline per GPU pipeline/subband
per mode (fast and slow).  For the expected number of GPU servers at OVRO-LWA this
equates to 32 pipeline instances.

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
    
  There are no optional arguments.  The command returns the base name for the files
  that will be written.
 * ``cancel`` - This cancels are previously scheduled or active recording.  The
    required arguments to this command are:
    
     * `queue_id` - an entry number in the recording queue to cancel.
     
   There are no optional arguments.  The command returns the base name for the files
   that were to have been written/were written that is associated with the queue
   entry.
 * ``delete`` - This deletes a file from the recording directory.  The required
   arguments to this command are:
   
    * ``file_number`` - an entry number in the file list of the recording directory.
    
  There are no optional arguments.  The command returns the name of the file that
  was deleted.

Monitoring Points
-----------------

There are several monitoring points for the pipeline.  All monitoring points live
under the etcd key "/mon/drv[sf]#", where [sf] indicates slow (s) or fast (f) mode
and # is the GPU pipeline that is being recorded.
  
  * /mon/drv[sf]#/bifrost
  
   * pipeline_lag - The lag between the system time and the
     timestamps for data in the pipeline.
   * max_acquire - The maximum span/gulp acquire time across
     all blocks in the pipeline.
   * max_process - The maximum span/gulp processing time
     across all blocks in the pipeline.
   * max_reserve - The maximum span/gulp reserve time across
     all blocks in the pipeline.
   * rx_rate - Packet capture rate for the pipeline.
   * rx_missing - Fraction of missing packets for the pipeline.
   
  * /mon/drv[sf]#/storage
 
   * active_disk_size - The size of the disk where the
     recording directory resides.
   * active_disk_free - The amount of free space on the disk
     where the recording directory resides.
   * active_directory - The current recording directory.
   * active_directory_size - The size of all files in the
     recording directory.
   * active_directory_count - The number of files in the
     recording directory.
   * files
   
     * files/name_# - The filename of the #-th entry in the
       recording directory.
     * files/size_# - The size of the #-th entry in the
       recording directory.
       
   * active_file - The name of the most recently created file
     in the recording directory.
   * active_file_size - The size of the most recently created
     file in the recording directory.
     
  * /mon/drv[sf]#/summary - An overall status of the pipeline.  Possible values
    are "normal", "warning", and "error".
  * /mon/drv[sf]#/info - A more detailed explanation of the summary condition.
  * /mon/drv[sf]#/diagnostics/
  
    * spectra - A URL-safe Base64 encoded PNG image of
      the auto-correlation spectra.
    * baselines - A URL-safe Base64 encoded PNG image
      of the visibility amplitude as a function of `(u,v)` radial distance.
      
  * /mon/drv[sf]#/statistics
    
    * min - A list of minimum values, one per antenna, from the auto-correlations.
    * avg - A list of mean values, one per antenna, from the auto-correlations.
    * max - A list of maximum values, one per antenna, from the auto-correlations.
     
Data Format
-----------

The pipeline writes out version 2 measurement sets.
