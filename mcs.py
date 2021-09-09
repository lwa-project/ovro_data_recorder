from __future__ import division, print_function
try:
    range = xrange
except NameError:
    pass
    
import os
import sys
import json
import time
import uuid
import etcd3
import base64
import signal
try:
    from io import BytesIO
except ImportError:
    from StringIO import StringIO as BytesIO
from datetime import datetime, timedelta
from textwrap import fill as tw_fill

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from common import ETCD_HOST, ETCD_PORT


__all__ = ['MonitorPoint', 'MultiMonitorPoint', 'ImageMonitorPoint',
           'MonitorPointCallbackBase', 'CommandCallbackBase', 'Client']


temp = etcd3.client()
ETCD_TYPE = type(temp)
del temp


class MonitorPoint(object):
    """
    Object for representing a monitoring point within the MCS framework.  At a
    minimum this includes:
    
     * a UNIX timestamp of when the monitoring point was updated,
     * the value of the monitoring point itself, and
     * the units of the monitoring point value or '' if there are none.
    """
    
    _required = ('timestamp', 'value', 'unit')
    
    def __init__(self, value, timestamp=None, unit='', **kwargs):
        if isinstance(value, MonitorPoint):
            value = value.as_dict()
            
        if isinstance(value, dict):
            for key in self._required:
                if key not in value:
                    raise KeyError("Missing required key: %s" % key)
        else:
            if timestamp is None:
                timestamp = time.time()
                
            value = {'timestamp': timestamp, 'value': value, 'unit': unit}
            for k,v in kwargs.items():
                value[k] = v
                
        self._entries = []
        for k,v in value.items():
            self._entries.append(k)
            setattr(self, k, v)
            
    def __repr__(self):
        output = "<%s " % (type(self).__name__,)
        first = True
        for k in self._entries:
            if not first:
                output += ', '
                
            v = getattr(self, k, None)
            if k == 'timestamp':
                v = datetime.utcfromtimestamp(v)
                v = str(v)
            if isinstance(v, str):
                v = "'%s'" % v
                
            output += "%s=%s" % (k, v)
            first = False
        output += '>'
        return tw_fill(output, subsequent_indent='    ')
        
    def __str__(self):
        output = "%s%s at %s" % (str(self.value), self.unit, datetime.utcfromtimestamp(self.timestamp))
        return output
        
    def __contains__(self, value):
        return value in self._entries
        
    @classmethod
    def from_json(cls, json_value):
        """
        Return a new :py:class:`MonitorPoint` instance based on a JSON-packed dictionary.
        """
        
        value = json.loads(json_value)
        return cls(value)
        
    def as_dict(self):
        """
        Return the information about the monitoring point as a dictionary.
        """
        
        value = {}
        for k in self._entries:
            value[k] = getattr(self, k)
        return value
        
    def as_json(self):
        """
        Return the information about the monitoring point as JSON-packed
        dictionary.
        """
        
        value = self.as_dict()
        return json.dumps(value)


class MultiMonitorPoint(MonitorPoint):
    """
    Sub-class of :py:class`MonitorPoint` for representing a multiple monitoring
    points updated at the same time within the MCS framework.  At a minimum this
    includes:
    
     * a UNIX timestamp of when the monitoring point was updated,
     * a list of values of the monitoring points themselves,
     * a list of field names for each monitoring point, and
     * the units of the monitoring point values or '' if there are none.
    
    .. note:: If only a single unit is supplied, it is replicated for all
              values.
    """
    
    _required = ('timestamp', 'value', 'field', 'unit')
    
    def __init__(self, value, timestamp=None, field=None, unit='', **kwargs):
        MonitorPoint.__init__(self, value, timestamp=timestamp, unit=unit, **kwargs)
        
        # The "multi" part
        try:
            len(self.value)
            if isinstance(self.value, (str, bytes)):
                raise TypeError
        except TypeError:
            self.value = [self.value,]
        if field is None:
            try:
                field = self.field
            except AttributeError:
                field = ['value%i' % i for i in range(len(self.value))]
        if isinstance(self.unit, str):
            self.unit = [self.unit for i in range(len(self.value))]
        if len(self.value) != len(field):
            raise RuntimeError("value and field keys must have the same length")
        if len(self.value) != len(self.unit):
            raise RuntimeError("value and unit keys must have the same length")
            
        if 'field' not in self._entries:
            self._entries.append('field')
        self.field = field
        
    def __str__(self):
        output = ', '.join(["%s=%s%s" % v for v in zip(self.field, self.value, self.unit)])
        output += " at %s" % datetime.utcfromtimestamp(self.timestamp)
        return output
        
    def as_list(self):
        """
        Return the information about the monitoring point as a list of three-
        element tuples containing (field name, value, unit).
        """
        
        return [e for e in zip(self.field, self.value, self.unit)]


class ImageMonitorPoint(MonitorPoint):
    """
    Sub-class of :py:class:`MonitorPoint` for representing a monitoring point
    image within the MCS framework.  At a minimum this includes:
    
     * a UNIX timestamp of when the monitoring point was updated,
     * the base64-encoded monitoring point image itself,
     * the MIME-type of the encoded image, and
     * the units of the monitoring point value or '' if there are none.
    """
    
    _required = ('timestamp', 'value', 'mime', 'unit')
    
    @staticmethod
    def _encode_image_data(image_data):
        image_data = base64.urlsafe_b64encode(image_data)
        try:
            image_data = image_data.decode()
        except AttributeError:
            pass
        return image_data
        
    @staticmethod
    def _decode_image_data(image_data):
        try:
            image_data = image_data.encode()
        except AttributeError:
            pass
        image_data = base64.urlsafe_b64decode(image_data)
        return image_data
        
    @classmethod
    def from_figure(cls, fig):
        """
        Return a new :py:class:`ImageMonitorPoint` instance based on a matplotlib Figure.
        """
        
        canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
        image = BytesIO()
        canvas.print_png(image)
        image.seek(0)
        image_data = image.read()
        image.close()
        
        image_data = cls._encode_image_data(image_data)
        
        return cls(image_data, mime='image/png')
        
    @classmethod
    def from_image(cls, im):
        """
        Return a new :py:class:`ImageMonitorPoint` instance based on a PIL.Image.
        """
        
        image = BytesIO()
        im.save(image, 'PNG')
        image.seek(0)
        image_data = image.read()
        image.close()
        
        image_data = cls._encode_image_data(image_data)
        
        return cls(image_data, mime='image/png')
        
    @classmethod
    def from_file(cls, name_or_handle):
        """
        Return a new :py:class:`ImageMonitorPoint` instance based the contents
        of a filename or open file handle.
        """
        
        try:
            ts = os.path.getmtime(name_or_handle.name)
            ext = os.path.getext(name_or_handle.name)[1]
            if ext not in ('.png', '.jpg', '.jpeg'):
                raise RuntimeError("Provided file does not seem to be a support image format")
                
            image_data = name_or_handle.read()
        except AttributeError:
            ts = os.path.getmtime(name_or_handle)
            ext = os.path.getext(name_or_handle)[1]
            if ext not in ('.png', '.jpg', '.jpeg'):
                raise RuntimeError("Provided file does not seem to be a support image format")
                
            with open(name_or_handle, 'rb') as fh:
                image_data = fh.read()
                
        image_data = cls._encode_image_data(image_data)
        
        mime = 'image/png'
        if ext in ('.jpg', '.jpeg'):
            mime = 'image/jpeg'
            
        return cls(image_data, timestamp=ts, mime=mime)
        
    def as_array(self):
        """
        Return the data for the monitoring point image as a numpy.array, similar
        to matplotlib.pyplot.imread.
        """
        
        image_data = self._decode_image_data(self.value)
        
        image = BytesIO()
        image.write(image_data)
        image.seek(0)
        
        image_data = plt.imread(image)
        image.close()
        return image_data
        
    def to_file(self, name_or_handle):
        """
        Write the monitoring point image to the specified filename or open
        file handle.
        """
        
        image_data = self._decode_image_data(self.value)
        
        try:
            name_or_handle.write(image_data)
        except AttributeError:
            with open(name_or_handle, 'wb') as fh:
                fh.write(image_data)


class MonitorPointCallbackBase(object):
    """
    Base class to use as a callback for when a monitoring point is changed.
    """
    
    @staticmethod
    def action(value):
        """
        Static method that should be overridden by the sub-class as to what to
        do with the new monitoring point value.  The method should accept a 
        single argument of a MonitorPoint.
        """
        
        raise NotImplementedError
        
    def __call__(self, event):
        output = None
        for evt in event.events:
            value = MonitorPoint.from_json(evt.value)
            output = self.action(value)
        return output


class CommandCallbackBase(object):
    """
    Base class to use as a callback for processing a command when it is
    received.
    """
    
    def __init__(self, client):
        if isinstance(client, Client):
            client = client.client
        elif isinstance(client, ETCD_TYPE):
            pass
        else:
            raise TypeError("Expected a mcs.Client or etcd3.client.Ectd3Client")
        self.client = client
        
    @staticmethod
    def action(*args, **kwargs):
        """
        Static method that should be overridden by the sub-class as to how to
        process the command.  The method should accept arbitrary keywords per
        the command and return a two-element tuple of (command status,
        response code).
        """
        
        raise NotImplementedError
        return status, response
        
    def __call__(self, event):
        for evt in event.events:
            try:
                key = evt.key.decode()
            except AttributeError:
                key = evt.key
            value = json.loads(evt.value)
            
            sequence_id = value['sequence_id']
            command = value['command']
            payload = value['kwargs']
            if 'sequence_id' not in payload:
                try:
                    payload['sequence_id'] = sequence_id.decode()
                except AttributeError:
                    payload['sequence_id'] = sequence_id
                    
            ts = time.time()
            status, response = self.action(**payload)
            if isinstance(status, bool):
                status = 'success' if status else 'error'
            status = {'sequence_id': sequence_id,
                      'timestamp': ts,
                      'status': status,
                      'response': response}
            status = json.dumps(status)
            
            key = '/resp/'+key[5:]
            self.client.put(key, status)


class Client(object):
    """
    MCS framework client.  This can be used for both monitor and control.
    """
    
    def __init__(self, id=None, timeout=5.0):
        """
        Initialize the client with the specified subsystem ID name and command
        response timeout in second.  A timeout of 'None' allows blocking until a
        response is received.  If the client is anonymous, i.e., 'id' is 'None'
        then only reading monitoring points and sending commands are supported.
        """
        
        if id is not None:
            id = str(id)
        self.id = id
        if timeout is None:
            timeout = 1e9
        self.timeout = timeout
        
        self.client = etcd3.client(host=ETCD_HOST, port=ETCD_PORT)
        self._mon_manifest = ['manifest/monitoring', 'manifest/commands']
        self._cmd_manifest = []
        self._watchers = {}
        
    def __del__(self):
        for command in self._watchers:
            try:
                self.client.cancel_watch(self._watchers[command][0])
            except Exception:
                pass
                
    def _update_mon_manifest(self, name, drop=False):
        """
        Update the monitoring point manifest as needed.  Returns a Boolean of
        whether or not an update was made.
        """
       
        with self.client.lock(self.id, ttl=5) as lock:
            # Is it alread in the local manifest?
            updated = False
            value = None
            if drop:
                ## Remove from the local manifest
                try:
                    del self._mon_manifest[self._mon_manifest.index(name)]
                except ValueError:
                    pass
                    
                ## Check the published manifest
                value = self.read_monitor_point('manifest/monitoring')
                if value is None:
                    value = MonitorPoint([])
                try:
                    del value.value[value.value.index(name)]
                    updated = True
                except ValueError:
                    pass
                    
            elif name not in self._mon_manifest:
                ## Not in the local manifest
                self._mon_manifest.append(name)
                
                ## Check the published manifest
                value = self.read_monitor_point('manifest/monitoring')
                if value is None:
                    value = MonitorPoint([])
                for entry in self._mon_manifest:
                    if entry not in value.value:
                        value.value.append(entry)
                        updated = True
                        
            # If there is an update, push it out
            if updated and value is not None:
                value.timestamp = time.time()
                value = value.as_json()
                self.client.put('/mon/%s/%s' % (self.id, 'manifest/monitoring'), value)
                
        return updated
        
    def _update_cmd_manifest(self, name, drop=False):
        """
        Update the command manifest as needed.  Returns a Boolean of whether or
        not an update was made.
        """
       
        with self.client.lock(self.id, ttl=5) as lock:
            # Is it alread in the local manifest?
            updated = False
            value = None
            if drop:
                ## Remove from the local manifest
                try:
                    del self._cmd_manifest[self._cmd_manifest.index(name)]
                except ValueError:
                    pass
                    
                ## Check the published manifest
                value = self.read_monitor_point('manifest/commands')
                if value is None:
                    value = MonitorPoint([])
                try:
                    del value.value[value.value.index(name)]
                    updated = True
                except ValueError:
                    pass
                    
            elif name not in self._cmd_manifest:
                ## Not in the local manifest
                self._cmd_manifest.append(name)
                
                ## Check the published manifest
                value = self.read_monitor_point('manifest/commands')
                if value is None:
                    value = MonitorPoint([])
                for entry in self._cmd_manifest:
                    if entry not in value.value:
                        value.value.append(entry)
                        updated = True
                        
            # If there is an update, push it out
            if updated and value is not None:
                value.timestamp = time.time()
                value = value.as_json()
                self.client.put('/mon/%s/%s' % (self.id, 'manifest/commands'), value)
                
        return updated
        
    def remove_monitor_point(self, name):
        """
        Remove the specified monitoring point.  Returns True if the deletion was
        successful, False otherwise.
        """
        
        if self.id is None:
            raise RuntimeError("Writing monitoring points is not supported in anonymous mode")
        if name.startswith('/'):
            name = name[1:]
            
        try:
            self.client.delete('/mon/%s/%s' % (self.id, name))
            self._update_mon_manifest(name, drop=True)
            return True
        except Exception as e:
            return False
            
    def write_monitor_point(self, name, value, timestamp=None, unit=''):
        """
        Write a value to the specified monitoring point.  Returns True if the
        write was successful, False otherwise.
        """
        
        if self.id is None:
            raise RuntimeError("Writing monitoring points is not supported in anonymous mode")
        if name.startswith('/'):
            name = name[1:]
        if isinstance(value, dict):
            pass
        elif isinstance(value, MonitorPoint):
            value = value.as_dict()
        else:
            if timestamp is None:
                timestamp = time.time()
                
            value = {'timestamp': timestamp, 'value': value, 'unit': unit}
        value = json.dumps(value)
        
        try:
            self.client.put('/mon/%s/%s' % (self.id, name), value)
            self._update_mon_manifest(name)
            return True
        except Exception:
            return False
            
    def read_monitor_point(self, name, id=None):
        """
        Read the current value of a monitoring point.  An 'id' of 'None' is
        interpretted as that monitoring point on the current subsystem.  Returns
        the monitoring point as a :py:class:`MonitorPoint` if successful, None
        otherwise.
        """
        
        if name.startswith('/'):
            name = name[1:]
        if id is None:
            id = self.id
        if id is None:
            raise RuntimeError("Must specify a subsystem ID when in anonymous mode")
            
        try:
            value = self.client.get('/mon/%s/%s' % (id, name))
            value = MonitorPoint.from_json(value[0])
            return value
        except Exception as e:
            print('ERROR2:', str(e))
            return None
            
    def list_monitor_points(self, id=None):
        """
        Return the contents of the monitoring point manifest.  An 'id' of 'None'
        is interpretted as that monitoring point on the current subsystem.  Returns
        the list if successful, None otherwise.
        """
        
        res = self.read_monitor_point('manifest/monitoring', id=id)
        if res is not None:
            res = res.value
        return res
        
    def set_monitor_point_callback(self, name, callback, id=None):
        """
        Watch the specified monitoring point and execute the callback when its
        value is updated.  The callback should be a sub-class of
        :py:class:`MonitorPointCallbackBase`.  An 'id' of 'None' is interpretted
        as that monitoring point on the current subsystem.  Return True is
        successful, False otherwise.  This watch...callback behavior continues
        until the appropriate :py:meth:`cancel_monitor_point_callback` is called.
        """
        
        if name.startswith('/'):
            name = name[1:]
        if not isinstance(callback, MonitorPointCallbackBase):
            raise TypeError("Expected a MonitorPointCallbackBase-derived instance")
        if id is None:
            id = self.id
        if id is None:
            raise RuntimeError("Must specify a subsystem ID when in anonymous mode")
            
        full_name = '/mon/%s/%s' % (id, name)
        try:
            watch_id = self.client.add_watch_callback(full_name, callback)
            try:
                self.client.cancel_watch(self._watchers[full_name][0])
            except KeyError:
                pass
            self._watchers[full_name] = (watch_id, callback)
            return True
        except Exception as e:
            return False
            
    def cancel_monitor_point_callback(self, name, id=None):
        """
        Cancel watching a monitoring point setup with set_monitor_point_callback
        Return True if successful, False otherwise.
        """
        
        if name.startswith('/'):
            name = name[1:]
        if id is None:
            id = self.id
        if id is None:
            raise RuntimeError("Must specify a subsystem ID when in anonymous mode")
            
        full_name = '/mon/%s/%s' % (id, name)
        try:
            self.client.cancel_watch(self._watchers[full_name][0])
            del self._watchers[full_name]
            return True
        except KeyError:
            return False
            
    def read_monitor_point_branch(self, name, id=None):
        """
        Read the current value of all keys in a monitoring point branch.  An
        'id' of 'None' is interpretted as that monitoring point branch on the
        current subsystem.  Returns the monitoring point branch as a list of
        two-element tuples of (key, :py:class:`MonitorPoint`) if successful, None
        otherwise.
        """
        
        if name.startswith('/'):
            name = name[1:]
        if id is None:
            id = self.id
        if id is None:
            raise RuntimeError("Must specify a subsystem ID when in anonymous mode")
            
        try:
            output = []
            for value in self.client.get_prefix('/mon/%s/%s' % (id, name)):
                output.append((value[1].key, MonitorPoint.from_json(value[0])))
            return output
        except Exception as e:
            print('ERROR3:', str(e))
            return None
            
    def set_monitor_point_branch_callback(self, name, callback, id=None):
        """
        Watch the specified monitoring point branch and execute the callback
        when any key within that branch is updated.  The callback should be a
        sub-class of :py:class:`MonitorPointCallbackBase`.  An 'id' of 'None' is
        interpretted as that monitoring point branch on the current subsystem.
        Return True is successful, False otherwise.  This watch...callback
        behavior continues until the appropriate
        :py:meth:`cancel_monitor_point_branch_callback` is called.
        """
        
        if name.startswith('/'):
            name = name[1:]
        if not isinstance(callback, MonitorPointCallbackBase):
            raise TypeError("Expected a MonitorPointCallbackBase-derived instance")
        if id is None:
            id = self.id
        if id is None:
            raise RuntimeError("Must specify a subsystem ID when in anonymous mode")
            
        full_name = '/mon/%s/%s' % (id, name)
        try:
            watch_id = self.client.add_watch_prefix_callback(full_name, callback)
            try:
                self.client.cancel_watch(self._watchers[full_name][0])
            except KeyError:
                pass
            self._watchers[full_name] = (watch_id, callback)
            return True
        except Exception as e:
            return False
            
    def cancel_monitor_point_branch_callback(self, name, id=None):
        """
        Cancel watching a monitoring point branch setup with
        :py:meth:`set_monitor_point_branch_callback`.  Return True if successful,
        False otherwise.
        """
        
        return self.canel_monitor_point_callback(name, id)
        
    def list_commands(self, subsystem):
        """
        Return the contents of the command manifest for the specified subsystem.
        Returns the list if successful, None otherwise.
        """
        
        res = self.read_monitor_point('manifest/commands', id=subsystem)
        if res is not None:
            res = res.value
        return res
            
    def set_command_callback(self, command, callback):
        """
        Process a command by executing the callback when it is received.  The
        callback should be a sub-class of :py:class:`CommandCallbackBase`.  Return
        True is successful, False otherwise.  This watch...callback behavior
        continues until the appropriate :py:meth:`cancel_command_callback` is
        called.
        """
        
        if self.id is None:
            raise RuntimeError("Command processing is not supported in anonymous mode")
        if command.startswith('/'):
            command = command[1:]
        if not isinstance(callback, CommandCallbackBase):
            raise TypeError("Expected a CommandCallbackBase-derived instance")
            
        full_name = '/cmd/%s/%s' % (self.id, command)
        try:
            watch_id = self.client.add_watch_prefix_callback(full_name, callback)
            try:
                self.client.cancel_watch(self._watchers[command][0])
            except KeyError:
                pass
            self._watchers[full_name] = (watch_id, callback)
            self._update_cmd_manifest(command)
            return True
        except Exception as e:
            return False
            
    def cancel_command_callback(self, command):
        """
        Cancel command processing setup with :py:meth:`set_command_callback`.  Return True
        if successful, False otherwise.
        """
        
        if self.id is None:
            raise RuntimeError("Command processing is not supported in anonymous mode")
        if command.startswith('/'):
            command = command[1:]
            
        full_name = '/cmd/%s/%s' % (self.id, command)
        try:
            self.client.cancel_watch(self._watchers[full_name][0])
            del self._watchers[full_name]
            self._update_cmd_manifest(command, drop=True)
            return True
        except KeyError:
            return False
            
    def send_command(self, subsystem, command, **kwargs):
        """
        Send a command to the given subsystem and wait for a response.  The 
        arguments for the command are given as keywords.  If a response is
        received within the timeout window, that response is returned as a two-
        element tuple of (True, the response as a dictionary).  If a response
        was not received within the timeout window or another error occurred,
        return a two-element tuple of (False, sequence_id).
        """
        
        if command.startswith('/'):
            command = command[1:]
            
        full_name = '/cmd/%s/%s' % (subsystem, command)
        resp_name = '/resp/'+full_name[5:]
        sequence_id = uuid.uuid1().hex
        try:
            s_id = sequence_id.decode()
        except AttributeError:
            s_id = sequence_id
        payload = {'sequence_id': sequence_id,
                   'timestamp': time.time(),
                   'command': command,
                   'kwargs': kwargs}
        payload = json.dumps(payload)
        
        def _timeout(signum, frame):
            raise RuntimeError("timeout")
        signal.signal(signal.SIGALRM, _timeout)

        try:
            if self.timeout > 0:
                signal.alarm(int(round(self.timeout)))
            events_iterator, cancel = self.client.watch(resp_name)
            
            self.client.put(full_name, payload)
            
            found = None
            t0 = time.time()
            while not found and (time.time() - t0) < self.timeout:
                for event in events_iterator:
                    value = json.loads(event.value)
                    if value['sequence_id'] == sequence_id:
                        found = value
                        break
            cancel()
            signal.alarm(0)
            
            return True, found
        except Exception as e:
            print('ERROR1:', str(e))
            return False, s_id
