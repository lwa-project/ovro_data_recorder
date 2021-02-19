import os
import sys
import signal
import logging
import threading
from datetime import datetime, timedelta
from logging.handlers import TimedRotatingFileHandler

from astropy.time import Time as AstroTime


__all__ = ['FS', 'CLOCK', 'NCHAN', 'CHAN_BW', 'NPIPELINE', 'OVRO_EPOCH',
           'LWATime', 'chan_to_freq', 'freq_to_chan', 'daemonize',
           'LogFileHandler', 'setup_signal_handling', 'synchronize_time']


FS               = 196.0e6
CLOCK            = 196.0e6
NCHAN            = 4096
CHAN_BW          = CLOCK / (2*NCHAN)
NPIPELINE        = 16
OVRO_EPOCH       = datetime(1970, 1, 1, 0, 0, 0, 0)


class LWATime(AstroTime):
    """
    Subclass of astropy.time.Time that accept a LWA timetag as an input.  It
    also attributes for accessing a (seconds, fraction seconds) tuple ("tuple")
    and a LWA timetag ("timetag").
    """
    
    def __init__(self, val, val2=None, format=None, scale=None, location=None, copy=False):
        if format == 'timetag':
            sec = val // int(FS)
            frac = (val % int(FS)) / FS
            val, val2 = sec, frac
            format = 'unix'
            if scale is not None:
                assert(scale.lower() == 'utc')
            else:
                scale = 'utc'
        AstroTime.__init__(self, val, val2, format=format, scale=scale,
                           precision=None, in_subfmt=None,
                           out_subfmt=None, location=location, copy=copy)
        
    @property
    def tuple(self):
        """
        Return a (seconds, fractional seconds) tuple.
        """
        
        val = self.timetag
        sec = val // int(FS)
        frac = (val % int(FS)) / FS
        return sec, frac
        
    @property
    def timetag(self):
        """
        Return a LWA timetag (ticks of a FS clock since the unix epoch).
        """
        
        sec1 = (self.jd1 - 2440588.0)*86400
        sec2 = self.jd2*86400
        sec = int(sec1) + int(sec2)
        frac = (sec1) - int(sec1) + sec2 - int(sec2)
        return int(sec*FS) + int(frac*FS)
        
    @property
    def casa_epoch(self):
        """
        Return a two element tuple of (CASA time reference, CASA time string)
        that is suitable for use with casacore.measures.measures.epoch.
        """
        
        dt = self.utc.datetime
        _, frac = self.tuple
        casa_time = '%i-%i-%i-%i:%i:%f' % (dt.year, dt.month, dt.day,
                                           dt.hour, dt.minute, dt.second+frac)
        return 'UTC', casa_time


def chan_to_freq(chan):
    """
    Convert a channel number to a frequency in Hz.
    """
    
    return CHAN_BW*(chan + 0.5)


def freq_to_chan(freq):
    """
    Convert a frequency in Hz to a channel number.
    """
    
    return int(freq / CHAN_BW + 0.5)


"""
This function is used to fork the current process into a daemon.
Almost none of this is necessary (or advisable) if your daemon
is being started by inetd. In that case, stdin, stdout and stderr are
all set up for you to refer to the network connection, and the fork()s
and session manipulation should not be done (to avoid confusing inetd).
Only the chdir() and umask() steps remain as useful.

From:
http://code.activestate.com/recipes/66012-fork-a-daemon-process-on-unix/

References:
UNIX Programming FAQ
    1.7 How do I get my program to act like a daemon?
        http://www.erlenstar.demon.co.uk/unix/faq_2.html#SEC16
        
    Advanced Programming in the Unix Environment
    W. Richard Stevens, 1992, Addison-Wesley, ISBN 0-201-56317-7.
"""

def daemonize(stdin='/dev/null', stdout='/dev/null', stderr='/dev/null'):
    """
    This forks the current process into a daemon.
    The stdin, stdout, and stderr arguments are file names that
    will be opened and be used to replace the standard file descriptors
    in sys.stdin, sys.stdout, and sys.stderr.
    These arguments are optional and default to /dev/null.
    Note that stderr is opened unbuffered, so
    if it shares a file with stdout then interleaved output
    may not appear in the order that you expect.
    """
    
    # Do first fork.
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0) # Exit first parent.
    except OSError as e:
        sys.stderr.write("fork #1 failed: (%d) %s\n" % (e.errno, e.strerror))
        sys.exit(1)
        
    # Decouple from parent environment.
    os.chdir("/")
    os.umask(0)
    os.setsid()
    
    # Do second fork.
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0) # Exit second parent.
    except OSError as e:
        sys.stderr.write("fork #2 failed: (%d) %s\n" % (e.errno, e.strerror))
        sys.exit(1)
        
    # Now I am a daemon!
    
    # Redirect standard file descriptors.
    si = file(stdin, 'r')
    so = file(stdout, 'a+')
    se = file(stderr, 'a+', 0)
    os.dup2(si.fileno(), sys.stdin.fileno())
    os.dup2(so.fileno(), sys.stdout.fileno())
    os.dup2(se.fileno(), sys.stderr.fileno())


class LogFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, rollover_callback=None):
        days_per_file =  1
        file_count    = 21
        TimedRotatingFileHandler.__init__(self, filename, when='D',
                                          interval=days_per_file,
                                          backupCount=file_count)
        self.filename = filename
        self.rollover_callback = rollover_callback
    def doRollover(self):
        super(LogFileHandler, self).doRollover()
        if self.rollover_callback is not None:
            self.rollover_callback()


def _handle_signal(signum, frame, log=None, threads=None, event=None):
    if log is None:
        log = logging.getLogger("__main__")
    SIGNAL_NAMES = dict((k, v) for v, k in \
                            reversed(sorted(signal.__dict__.items()))
                            if v.startswith('SIG') and \
                            not v.startswith('SIG_'))
    log.warning("Received signal %i %s", signum, SIGNAL_NAMES[signum])
    try:
        threads[0].shutdown()
    except (TypeError, IndexError):
        pass
    try:
        event.set()
    except TypeError:
        pass


def setup_signal_handling(threads, log=None, signals=[signal.SIGHUP,
                                                      signal.SIGINT,
                                                      signal.SIGQUIT,
                                                      signal.SIGTERM,
                                                      signal.SIGTSTP]):
    shutdown_event = threading.Event()
    def handler(signum, frame):
        _handle_signal(signum, frame, log=log, threads=threads, event=shutdown_event)
    for sig in signals:
        signal.signal(sig, handler)
    return shutdown_event


def synchronize_time(server='ntp.ubuntu.com'):
    success = False
    try:
        subprocess.check_call(['ntpdate', server],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        success = True
    except subprocess.CalledProcessError:
        pass
    return sucess
