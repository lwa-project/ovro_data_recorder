dr_vbeam.py
=============

``dr_vbeam.py`` is an alternative to ``dr_tengine.py`` that captures packetized
beam data from the digital system, down selects the frequency coverage to match
what is requested by the observer, and write it directly to disk in RBeam files.

Structure
---------

The pipeline is written in the Bifrost framework and has four blocks:  
``CaptureOp``, ``DownSelectOp``, and ``RawWriterOp``.

 * ``CaptureOp`` - This is the data capture block which is responsible for capturing
   the beam packets from the digital system, ordering them in time and frequency,
   and writing the organized data to a Bifrost ring.
 * ``DownSelectOp`` - This takes the full voltage beam bandwidth and downselects it
   to a frequency range provided by the observer.
 * ``RawWriterOp`` - This reads in down selected data data and writes raw RBeam files
   to disk.

 The pipeline is designed such that there is one pipeline per voltage beam.  For the
 expected number un-averaged beams created by the digital system this equates to 2
 pipeline instances.

Control Commands
----------------

The ``dr_vbeam.py`` pipeline supports six commands: ``ping``, ``sync``, ``raw_record``,
``cancel``, ``delete``, and ``bnd``.

  * ``ping`` - This command simply replies which is helpful to see if the pipeline
    is responsive.  There are no required or optional arguments.  Returns a response
    of "pong".
  * ``sync`` - This command sets the system time via NTP.  The required arguments to
    this command are:
  
     * ``server`` - a NTP server name or IP address to sync against.
   
    There are no optional arguments.  The command returns the sync status.
  * ``raw_record`` - This schedules a recording to take place.  The required arguments to
    this command are:
    
     * ``start_mjd`` - an integer MJD value for when the recording will start,
     * ``start_mpm`` - an integer number of milliseconds past midnight value on the
       MJD specified in ``start_mjd`` for when the recording will start, and
     * ``duration_ms`` - the number of milliseconds to record data for.
     
    There are no optional arguments.  The command returns the name of the file that
    will be written.  The name will be of the format "<mjd>_<MCS sequence id>".
  * ``cancel`` - This cancels a previously scheduled or active recording.  The
     required arguments to this command are:
     
      * `queue_number` - an entry number in the recording queue to cancel.
      
    There are no optional arguments.  The command returns the base name for the files
    that were to have been written/were written that is associated with the queue
    entry.
  * ``delete`` - This deletes a file from the recording directory.  The required
    arguments to this command are:
    
     * ``file_number`` - an entry number in the file list of the recording directory.
     
    There are no optional arguments.  The command returns the name of the file that
    was deleted.
  * ``bnd`` - This controls the down selection in frequency.  The required arguments to
    this command are:
    
     * ``beam`` - an integer of the voltage beam number of control,
     * ``central_freq`` - the central frequency of the down selection in Hz, and
     * ``bw`` - the bandwidth in Hz.
    
Monitoring Points
-----------------

There are several monitoring points for the pipeline.  All monitoring points live
under the etcd key "/mon/drr#", where # is the power beam number that is being
recorded.
  
  * /mon/drr#/bifrost
  
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
   
  * /mon/drr#/storage
 
    * active_disk_size - The size of the disk where the
      recording directory resides.
    * active_disk_free - The amount of free space on the disk
      where the recording directory resides.
    * active_directory - The current recording directory.
    * active_directory_size - The size of all files in the
      recording directory.
    * active_directory_count - The number of files in the
      recording directory.
  
  * /mon/drr#/summary - An overall status of the pipeline.  Possible values
    are "normal", "warning", and "error".
  * /mon/drr#/info - A more detailed explanation of the summary condition.
     


Data Format
-----------

The RBeam format is a packetized format for storing complex frequency domain
timeseries data.  The 16 B header for these packets is defined as:

.. csv-table:: Header Fields
  :header: Name, Data Type, Notes
  
  server,  uint8_t,  1-based
  gbe,     uint8_t,  not used
  nchan,   uint16_t, big endian
  nbeam,   uint8_t,  always 1
  nserver, uint8_t,  always 1
  chan0,   uint16_t, big endian; first channel in packet
  seq,     uint64_t, big endian; 1-based

Following this header is a data section composed of little endian packed
single precision floating point values, one for each beam, channel, and
both polarizations contained in the frame.  This is a 3D data structure with
axes beam x channel x polarization (X and Y).

Sequence numbers can be converted to UNIX time stamps, `t` via:

.. math::
  t = seq \times \frac{8192}{196e6}.
