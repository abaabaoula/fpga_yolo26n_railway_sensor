"""Bound-checked direct FP32 integer simulation, with an FP64 audit mode.

All operands and products are integers.  If sum(abs(products))+abs(bias)
is below 2**24, each partial sum in direct convolution is exactly representable
in FP32. MKLDNN/Winograd and reduced-precision paths are explicitly disabled.
This accelerates PC evaluation only; board arithmetic remains INT8/INT32.
"""
import numpy as np
import torch
import torch.nn.functional as F

class BoundedIntegerBackend:
    def __init__(self,shadow=False):
        self.weights={};self.shadow=shadow;self.calls=0;self.checked_elements=0;self.bounds={}
    def execute(self,x,w,b,spec,constants):
        sid=spec['id'];q=spec['quant'];zx=int(constants[q['input_zp']])
        if sid not in self.weights:
            wc=w.astype(np.int64)-constants[q['weight_zp']].astype(np.int64).reshape(-1,1,1,1)
            bound=255*np.abs(wc).reshape(w.shape[0],-1).sum(1)+np.abs(b.astype(np.int64))
            maximum=int(bound.max());self.bounds[sid]=maximum
            dtype=torch.float32 if maximum<2**24 else torch.float64
            if maximum>=2**53:raise RuntimeError('No exact floating-point integer simulation available')
            self.weights[sid]=(torch.from_numpy(wc).to(dtype),torch.from_numpy(b.astype(np.int64)).to(dtype))
        weight,bias=self.weights[sid]
        inp=torch.from_numpy(np.ascontiguousarray(x)).to(weight.dtype).sub_(zx)
        with torch.inference_mode(),torch.backends.mkldnn.flags(enabled=False):
            result=F.conv2d(inp,weight,bias,stride=spec['stride'],padding=spec['padding'],groups=spec['groups'])
            if self.shadow:
                reference=F.conv2d(inp.to(torch.float64),weight.to(torch.float64),bias.to(torch.float64),stride=spec['stride'],padding=spec['padding'],groups=spec['groups'])
                if not torch.equal(result.to(torch.float64),reference):
                    raise RuntimeError(('Integer accumulator audit failed',sid,int(torch.count_nonzero(result.to(torch.float64)!=reference))))
                self.checked_elements+=result.numel()
        value=result.numpy()[0]
        if not np.all(value==np.rint(value)) or value.min() < -2**31 or value.max()>=2**31:raise RuntimeError('Non-integer or overflowing accumulator')
        self.calls+=1
        return value.astype(np.int32)
