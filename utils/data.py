"""
Stub module for downloading the latest data.py from https://github.com/lwa-project/commissioning
"""

from __future__ import print_function

import os
import time
try:
    from urllib import request as urlrequest
except ImportError:
    import urllib2 as urlrequest


# URL to download
MODULE_URL = 'https://raw.githubusercontent.com/lwa-project/commissioning/master/DRX/HDF5/data.py'

# Where it lives
MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
MODULE_NAME = os.path.join(MODULE_PATH, '_data.py')
MODULE_ETAG = os.path.join(MODULE_PATH, '_data.etag')

# Maximum file age to accept without checking for a newer version
MAX_AGE_SEC = 86400


# Get the current file age and entity tag (if it exists)
age = 1e6
etag = ''
if os.path.exists(MODULE_NAME):
    age = time.time() - os.path.getmtime(MODULE_NAME)
    if os.path.exists(MODULE_ETAG):
        with open(MODULE_ETAG, 'r') as fh:
            etag = fh.read()
            
# If the file is more than MAX_AGE_SEC old, check for an update
if age > MAX_AGE_SEC:
    request = urlrequest.Request(MODULE_URL)
    opener = urlrequest.build_opener()
    data = opener.open(request)
    if data.headers['etag'] != etag:
        with open(MODULE_NAME, 'wb') as fh:
            fh.write(data.read())
        with open(MODULE_ETAG, 'w') as fh:
            fh.write(data.headers['etag'])
            
# Load in everything from the module
from _data import *
from _data import __all__
