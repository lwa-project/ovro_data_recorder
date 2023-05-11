dr_beam.py
==========

``dr_beam.py`` is the recording pipeline for the power beam mode.  This pipeline
captures packetized power beam data from the digital system and writes the data
to HDF5 files.  It can also perform on-the-fly transformations of the power beam
data to reduce the time or frequency or change the polarization products being
recorded.  In addition, the pipeline also generates diagnostic plots of the spectra.

Structure
---------

The pipeline is written in the Bifrost framework and has four blocks:  
``CaptureOp``, ``SpectraOp``, ``StatisticsOp``, and ``WriterOp``.

 * ``CaptureOp`` - This is the data capture block which is responsible for capturing
   the power beam packets from the digital system, ordering them in time and frequency,
   and writing the organized data to a Bifrost ring.
 * ``SpectraOp`` - This reads in the power beam data and writes a spectra to a PNG file
   every 60 s.
 * ``StatisticsOp`` - This reads in the power beam data and computes the
   minimum/mean/maximum values across time and frequency on a 60 s cadence.
 * ``WriterOp`` - This reads in the visibility data and writes HDF5 files to disk.

The pipeline is designed such that there is one pipeline per power beam.  For the
expected number power beams created by the digital system this equates to 12 
pipeline instances.

Control Commands
----------------

The ``dr_beam.py`` pipeline supports five commands: ``ping``, ``sync``, ``record``,
``cancel``, and ``delete``.

 * ``ping`` - This command simply replies which is helpful to see if the pipeline
   is responsive.  There are no required or optional arguments.  Returns a response
   of "pong".
 * ``sync`` - This command sets the system time via NTP.  The required arguments to
   this command are:

    * ``server`` - a NTP server name or IP address to sync against.
 
   There are no optional arguments.  The command returns the sync status.
 * ``record`` - This schedules a recording to take place.  The required arguments to
   this command are:
   
    * ``start_mjd`` - an integer MJD value for when the recording will start,
    * ``start_mpm`` - an integer number of milliseconds past midnight value on the
      MJD specified in ``start_mjd`` for when the recording will start, and
    * ``duration_ms`` - the number of milliseconds to record data for.
    
   The optional arguments are:
  
    * ``stokes_mode`` - a string of "XXYY", "CRCI", "IQUV", or "IV" that specifies
      what polarization products are to be recorded.  If not provided the native
      XXYYCRCI products are recorded.  `Note:  For IQUV and IV the quantities computed
      are psuedo-Stokes parameters.`
    * ``time_avg`` - an integer of the number of consecutive power beam spectra to
      average together when writing to the file.  Must be a power of two between 1
      and 1024.  If not specified no averaging in time is performed.
    * ``chan_avg`` - an integer of the number of consecutive channels in the power
      beam spectra to average together when writing to the file.  Must be an even
      divisor of 184, i.e., ``184 % chan_avg = 0``.  If not specified no averaging
      in frequency is performed.
    
   The command returns the name of the file that will be written.  The name will be
   of the format "<mjd>_<MCS sequence id>".
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

Monitoring Points
-----------------

There are several monitoring points for the pipeline.  All monitoring points live
under the etcd key "/mon/dr#", where # is the power beam number that is being
recorded.
  
  * /mon/dr#/bifrost
  
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
   
  * /mon/dr#/storage
 
    * active_disk_size - The size of the disk where the
      recording directory resides.
    * active_disk_free - The amount of free space on the disk
      where the recording directory resides.
    * active_directory - The current recording directory.
    * active_directory_size - The size of all files in the
      recording directory.
    * active_directory_count - The number of files in the
      recording directory.
     
  * /mon/dr#/summary - An overall status of the pipeline.  Possible values
    are "normal", "warning", and "error".
  * /mon/dr#/info - A more detailed explanation of the summary condition.
  * /mon/dr#/diagnostics/
  
    * spectra - A URL-safe Base64 encoded PNG image of
      the powe beam spectra.
      
  * /mon/dr#/statistics
    
    * min - A list of minimum values, one per polarization, for the beam.
    * avg - A list of mean values, one per polarization, for the beam.
    * max - A list of maximum values, one per polarization, for the beam.
     


Data Format
-----------

The HDF5 files written by the pipeline have the following structure:

 * <top level>
   
   * ObserverID - `attribute` - observer's ID number, default is 0
   * ObserverName - `attribute` - observer's name, default is blank
   * ProjectID - `attribute` - project ID, default is blank
   * SessionID - `attribute` - session ID, default is 0
   * StationName - `attribute` - name of the station where the data were recorded
   * FileCreation - `attribute` - File creation time
   * FileGenerator - `attribute` - Name of the softare that created the file
   * InputMetadata - `attribute` - Observation metadata file, default is blank
   * /Observation# - `group`
     
     * time - `data set` - times for each integration
     * TargetName - `attribute` - The name of the target being observed, default
       is blank
     * RA - `attribute` - The RA of the target, default is -99.0
     * RA_Units - `attribute` - The units of the "RA" field, default is hours
     * Dec - `attribute` - The dec. of the target, default is -99.0
     * Dec_Units - `attribute` - The units of the "Dec" field, default is degrees
     * Epoch - `attribute` - The epoch of the coordinates, default is 2000.0
     * TrackingMode - `attribute` - The tracking mode for the observation, default
       is 'Unknown'
     * ARX_Filter - `attribute` - ARX filter used for the observation, default is -1.0
     * ARX_Gain1 - `attribute` - ARX attenuation setting for the first attenuator,
       default is -1.0
     * ARX_Gain2 - `attribute` - ARX attenuation setting for the second attenuator,
       default is -1.0
     * ARX_GainS - `attribute` - ARX attenuation setting for the shelf attenuator,
       default is -1.0
     * Beam - `attribute` - Beam number used for the observation
     * DRX_Gain - `attribute` - Digital gain value for the observation, default
       is -1.0
     * sampleRate - `attribute` - Sample rate used for the observation
     * sampleRate_Units - `attribute` - The units of the "sampleRate" field,
       default is Hz
     * tInt - `attribute` - Integration time for the spectra, default is -1.0
     * tInt_Units - `attribute` - The units for the "tInt" field, default is s
     * LFFT - `attribute` - Number of FFT channels used to move to the frequency
       domain
     * nChan - `attribute` - The number of channels recorded to the file, default
       is 0
     * RBW - `attribute` - The resolution bandwidth of the recorded data, default
      is -1.0
     * RBW_Units - `attribute` - The units of the "RBW" field, default is Hz
     * /Tuning1
       
       * freq - data set - frequencies for each channel
       * <polarization_1> - data set - the time-frequency data for first polarization
         recorded, named by the name of the polarization product
       * ...
       * <polarization_N> - data set - the time-frequency data for last polarization
         recorded, named by the name of the polarization product
