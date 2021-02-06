import numpy


class ReductionOperation(object):
    def __init__(self, time_avg=1, chan_avg=1):
        self.time_avg = 1
        self.chan_avg = 1
        
        self.pols = 'XX,YY,CR,CI'
        self.reductions = (self.time_avg,1,self.chan_avg,1)
        
    def __repr__(self):
        return "<%s time_avg=%i, chan_avg=%i, pols='%s'>" % (type(self).__name__,
                                                             self.time_avg,
                                                             self.chan_avg,
                                                             self.pols)
        
    def __eq__(self, other):
        return (self.reductions == other.reductions) and (self.pols == other.pols)
        
    def __ne__(self, other):
        return not (self == other)
        
    def _average(self, idata):
        odata = idata
        if self.time_avg != 1:
            odata = idata.reshape(-1,self.time_avg,idata.shape[1],idata.shape[2],idata.shape[3])
            odata = numpy.mean(odata, axis=1)
        if self.chan_avg:
            odata = odata.reshape(odata.shape[0],odata.shape[1],-1,self.chan_avg,odata.shape[3])
            odata = numpy.mean(odata, axis=2)
        return odata
        
    def __call__(self, idata):
        odata = self._average(idata)
        return odata


FullLinear = ReductionOperation


class XXandYY(ReductionOperation):
    def __init__(self, time_avg=1, chan_avg=1):
        ReductionOperation.__init__(self, time_avg=time_avg, chan_avg=chan_avg)
        
        self.pols = 'XX,YY'
        self.reductions = (self.time_avg, 1, self.chan_avg, 2)
        
    def __call__(self, idata):
        odata = self._average(idata)
        return odata[...,[0,1]]


class CRandCI(ReductionOperation):
    def __init__(self, time_avg=1, chan_avg=1):
        ReductionOperation.__init__(self, time_avg=time_avg, chan_avg=chan_avg)
        
        self.pols = 'CR,CI'
        self.reductions = (self.time_avg, 1, self.chan_avg, 2)
        
    def __call__(self, idata):
        odata = self._average(idata)
        return odata[...,[2,3]]


class FullStokes(ReductionOperation):
    def __init__(self, time_avg=1, chan_avg=1):
        ReductionOperation.__init__(self, time_avg=time_avg, chan_avg=chan_avg)
        
        self.pols = 'I,Q,U,V'
        self.reductions = (self.time_avg, 1, self.chan_avg, 1)
        
    def __call__(self, idata):
        odata = sel._average(idata)
        # TODO: check
        I = odata[...,0] + odata[...,1]
        Q = odata[...,0] - odata[...,1]
        odata[...,0] = I
        odata[...,1] = Q
        odata[...,2] *= 2
        odata[...,3] *= 2
        return odata


class IandV(ReductionOperation):
    def __init__(self, time_avg=1, chan_avg=1):
        ReductionOperation.__init__(self, time_avg=time_avg, chan_avg=chan_avg)
        
        self.pols = 'I,V'
        self.reductions = (self.time_avg, 1, self.chan_avg, 2)
        
    def __call__(self, idata):
        odata = sel._average(idata)
        # TODO: check
        odata[...,0] += odata[...,1]
        odata[...,1] = 2*odata[...,3]
        return odata[...,[0,1]]
