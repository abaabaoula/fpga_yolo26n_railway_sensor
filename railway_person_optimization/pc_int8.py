"""Exact CPU INT8 GEMM simulator; never used as a board timing measurement."""
import numpy as np
import torch
import torch.nn.functional as F
from .pc_integer import BoundedIntegerBackend

class Int8IntegerBackend:
    def __init__(self,shadow=False):
        self.shadow=shadow;self.weights={};self.fallback=BoundedIntegerBackend(shadow)
        self.calls=0;self.checked_elements=0;self.bounds={}
    def execute(self,x,w,b,spec,constants):
        q=spec['quant'];sid=spec['id'];zx=int(constants[q['input_zp']]);wz=constants[q['weight_zp']]
        if spec['groups']!=1 or np.any(wz!=0):
            result=self.fallback.execute(x,w,b,spec,constants)
            self.calls+=1;self.checked_elements+=result.size if self.shadow else 0
            return result
        if sid not in self.weights:
            flat=w.reshape(w.shape[0],-1)
            correction=b.astype(np.int64)-zx*flat.astype(np.int64).sum(axis=1)
            bound=128*np.abs(flat.astype(np.int64)).sum(axis=1)+np.abs(correction)
            maximum=int(bound.max());self.bounds[sid]=maximum
            if maximum>=2**31:raise RuntimeError('INT8 GEMM accumulation bound exceeds INT32')
            self.weights[sid]=(torch.from_numpy(np.ascontiguousarray(flat.T)),correction.astype(np.int32))
        weight,correction=self.weights[sid]
        assert x.shape[0]==1 and x.dtype==w.dtype==np.int8
        pad=spec['padding'];k=spec['K'];stride=spec['stride']
        source=np.pad(x[0],((0,0),(pad,pad),(pad,pad)),constant_values=zx) if pad else x[0]
        windows=np.lib.stride_tricks.sliding_window_view(source,(k,k),axis=(1,2))[:,::stride,::stride]
        c,oh,ow,_,_=windows.shape
        matrix=np.ascontiguousarray(windows.transpose(1,2,0,3,4).reshape(oh*ow,c*k*k))
        with torch.inference_mode():
            accum=torch._int_mm(torch.from_numpy(matrix),weight).numpy()
            accum+=correction
            result=np.ascontiguousarray(accum.T.reshape(w.shape[0],oh,ow))
            if self.shadow:
                ref=F.conv2d(torch.from_numpy(np.ascontiguousarray(x)).to(torch.float64).sub_(zx),torch.from_numpy(w.astype(np.float64)),torch.from_numpy(b.astype(np.float64)),stride=stride,padding=pad).numpy()[0]
                if not np.array_equal(result,ref):raise RuntimeError(('Exact INT8 CPU accumulator mismatch',sid,int(np.count_nonzero(result!=ref))))
                self.checked_elements+=result.size
        self.calls+=1
        return result
