``dr_tengine.py``
=================

``dr_tengine.py`` is a combination T-engine and recording pipeline for the voltage
beam mode.  This pipeline captures packetized beam data from the digital system,
down selects the frequency coverage to match what is requested by the observer,
converts the data back into the time domain, and writes raw DRX files.

Structure
---------

The pipeline is written in the Bifrost framework and has four blocks:  
``CaptureOp``, ``ReChannelizerOp``, ``TEngineOp``, and ``WriterOp``.

 * ``CaptureOp`` - This is the data capture block which is responsible for capturing
   the beam packets from the digital system, ordering them in time and frequency,
   and writing the organized data to a Bifrost ring.
 * ``ReChannelizerOp`` - This reads in the beam data and performs an inverse FFT
   followed by an FFT to change the channel width of the data to 50 kHz.
 * ``TEngineOp`` - This reads in the re-channelized data, down selects the frequency
   range, and performs an inverse FFT to create time domain voltage data.
 * ``WriterOp`` - This reads in time domain data and writes raw DRX files to disk.

 The pipeline is designed such that there is one pipeline per voltage beam.  For the
 expected number un-averaged beams created by the digital system this equates to 2
 pipeline instances.

Control Commands
----------------

The ``dr_tengine.py`` pipeline supports four commands: ``record``, ``cancel``, 
``delete``, and ``drx``.

  * ``record`` - This schedules a recording to take place.  The required arguments to
    this command are:
    
     * ``sequence_id`` - a monitor and control system command ID (used for naming
       the output files),
     * ``start_mjd`` - an integer MJD value for when the recording will start,
     * ``start_mpm`` - an integer number of milliseconds past midnight value on the
       MJD specified in ``start_mjd`` for when the recording will start, and
     * ``duration_ms`` - the number of milliseconds to record data for.
     
    There are no optional arguments.  The command returns the base name for the
    files that will be written.
  * ``cancel`` - This cancels are previously scheduled or active recording.  The
     required arguments to this command are:
     
      * `sequence_id`` - a monitor and control system command ID and
      * `queue_id` - an entry number in the recording queue to cancel.
      
    There are no optional arguments.  The command returns the base name for the files
    that were to have been written/were written that is associated with the queue
    entry.
  * ``delete`` - This deletes a file from the recording directory.  The required
    arguments to this command are:
    
     * ``sequence_id`` - a monitor and control system command ID and 
     * ``file_number`` - an entry number in the file list of the recording directory.
     
    There are no optional arguments.  The command returns the name of the file that
    was deleted.
  * ``drx`` - This controls the down selection in frequency for the T-engine as
    as well as the quantization gain when moving to 4+4-bit complex integers for
    the output.  The required arguments to
    this command are:
    
     * ``sequence_id`` - a monitor and control system command ID,
     * ``beam`` - an integer of the voltage beam number of control,
     * ``tuning`` - an integer of the tuning number, 1 or 2, within the voltage
       beam to control, 
     * ``central_freq`` - the central frequency of the tuning in Hz,
     * ``filter`` - an integer for the output bandwidth code, 7 = 19.6 MHz, and
     * ``gain`` - an integer for the quantization gain.
     
    There is a single optional argument of ``subslot`` which controls when the
    command is implemented within the second.  If not specified the default of 0
    is used.
    
Monitoring Points
-----------------

There are several monitoring points for the pipeline.  All monitoring points live
under the etcd key "/mon/drt#", where # is the power beam number that is being
recorded.
  
  * /mon/drt#/bifrost
  
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
   
  * /mon/drt#/storage
 
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
     
  * /mon/drt#/summary - An overall status of the pipeline.  Possible values
    are "normal", "warning", and "error".
  * /mon/drt#/info - A more detailed explanation of the summary condition.
     


Data Format
-----------

Is the DRX format.  What more is there to say?
