import os
import sys
import signal
import logging
import threading


FS               = 196.0e6
CLOCK            = 196.0e6
NCHAN            = 4096
CHAN_BW          = CLOCK / (2*NCHAN)


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
This module is used to fork the current process into a daemon.
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


def _handle_signal(signum, frame, threads=None, event=None):
    log = logging.getLogger(__name__)
    SIGNAL_NAMES = dict((k, v) for v, k in \
                            reversed(sorted(signal.__dict__.items()))
                            if v.startswith('SIG') and \
                            not v.startswith('SIG_'))
    log.warning("Received signal %i %s", signum, SIGNAL_NAMES[signum])
    try:
        threads.shutdown()
    except (TypeError, IndexError):
        pass
    try:
        event.set()
    except TypeError:
        pass


def setup_signal_handling(threads, signals=[signal.SIGHUP,
           	                            signal.SIGINT,
                                            signal.SIGQUIT,
                                            signal.SIGTERM,
                                            signal.SIGTSTP]):
    shutdown_event = threading.Event()
    def handler(signum, frame):
        _handle_signal(signum, frame, threads=threads, event=shutdown_event)
    for sig in signals:
        signal.signal(sig, handler)
    return shutdown_event
